import cv2
import os
import json
import logging
import hashlib
import time
import sqlite3
from datetime import datetime, timedelta
import shutil

import config

logger = logging.getLogger(__name__)

class Incident:
    """Represents a recorded incident with metadata and video frames."""
    
    def __init__(self, incident_id, timestamp, risk_type, risk_score, location=None):
        """
        Initialize an incident.
        
        Args:
            incident_id: Unique ID for the incident
            timestamp: When the incident occurred
            risk_type: Type of risk (e.g., 'collision', 'lane_departure')
            risk_score: Risk score (0-1)
            location: Optional location data
        """
        self.incident_id = incident_id
        self.timestamp = timestamp
        self.risk_type = risk_type
        self.risk_score = risk_score
        self.location = location
        self.frames = []
        self.metadata = {}
        self.hash = None
    
    def add_frame(self, frame, detections=None, risks=None):
        """
        Add a frame to the incident.
        
        Args:
            frame: Image frame
            detections: List of detections in the frame
            risks: List of risks in the frame
        """
        frame_data = {
            'frame': frame,
            'detections': detections,
            'risks': risks,
            'timestamp': datetime.now()
        }
        
        self.frames.append(frame_data)
    
    def to_dict(self):
        """
        Convert incident to a dictionary for serialization.
        
        Returns:
            Dictionary representation of the incident
        """
        # Process metadata to ensure JSON serializable
        processed_metadata = {}
        for k, v in self.metadata.items():
            if k == 'alerts':
                processed_alerts = []
                for alert in v:
                    processed_alert = {}
                    for ak, av in alert.items():
                        if isinstance(av, (int, float, str, type(None))):
                            processed_alert[ak] = av
                        elif isinstance(av, bool):
                            processed_alert[ak] = av
                        else:
                            processed_alert[ak] = str(av)
                    processed_alerts.append(processed_alert)
                processed_metadata[k] = processed_alerts
            elif isinstance(v, (int, float, str, type(None))):
                processed_metadata[k] = v
            elif isinstance(v, bool):
                processed_metadata[k] = v
            else:
                processed_metadata[k] = str(v)
        
        return {
            'incident_id': self.incident_id,
            'timestamp': self.timestamp.isoformat(),
            'risk_type': self.risk_type,
            'risk_score': self.risk_score,
            'location': self.location,
            'metadata': processed_metadata,
            'hash': self.hash,
            'frame_count': len(self.frames)
        }
    
    @classmethod
    def from_dict(cls, data):
        """
        Create an incident from a dictionary.
        
        Args:
            data: Dictionary representation of an incident
            
        Returns:
            Incident object
        """
        incident = cls(
            incident_id=data['incident_id'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            risk_type=data['risk_type'],
            risk_score=data['risk_score'],
            location=data.get('location')
        )
        
        incident.metadata = data.get('metadata', {})
        incident.hash = data.get('hash')
        
        return incident

class IncidentManager:
    """
    Manages recording, storing, and retrieving incidents.
    """
    
    def __init__(self, data_dir=None, retention_days=None):
        """
        Initialize the incident manager.
        
        Args:
            data_dir: Directory to store incidents, or None to use the path from config
            retention_days: Number of days to retain incidents, or None to use the value from config
        """
        self.data_dir = data_dir or os.path.join(config.DATA_DIR, "incidents")
        self.retention_days = retention_days or config.INCIDENT_RETENTION_DAYS
        
        # Create incidents directory if it doesn't exist
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Initialize database
        self.db_path = os.path.join(self.data_dir, "incidents.db")
        self._init_database()
        
        logger.info(f"Incident manager initialized with data directory: {self.data_dir}")
        logger.info(f"Retention period: {self.retention_days} days")
    
    def _init_database(self):
        """Initialize the SQLite database for incident metadata."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create incidents table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS incidents (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT,
                    risk_type TEXT,
                    risk_score REAL,
                    location TEXT,
                    metadata TEXT,
                    hash TEXT,
                    frame_count INTEGER
                )
            ''')
            
            conn.commit()
            conn.close()
            
            logger.info("Incident database initialized")
        
        except Exception as e:
            logger.error(f"Failed to initialize incident database: {e}")
    
    def record_incident(self, timestamp, frame, detections=None, lanes=None, risks=None, alerts=None):
        """
        Record a new incident.
        
        Args:
            timestamp: When the incident occurred
            frame: Image frame
            detections: List of detections in the frame
            lanes: List of detected lanes
            risks: List of risks
            alerts: List of alerts
            
        Returns:
            Incident ID
        """
        try:
            # Generate incident ID
            incident_id = f"incident_{timestamp.strftime('%Y%m%d_%H%M%S')}_{int(time.time() * 1000) % 1000}"
            
            # Determine the highest risk and its type
            max_risk_score = 0
            max_risk_type = "unknown"
            
            if risks and len(risks) > 0:
                for risk in risks:
                    if risk.risk_score > max_risk_score:
                        max_risk_score = risk.risk_score
                        max_risk_type = risk.risk_type
            
            # Create incident
            incident = Incident(
                incident_id=incident_id,
                timestamp=timestamp,
                risk_type=max_risk_type,
                risk_score=max_risk_score
            )
            
            # Add frame
            incident.add_frame(frame, detections, risks)
            
            # Add metadata
            if alerts:
                incident.metadata['alerts'] = [
                    {
                        'message': alert.message,
                        'explanation': alert.explanation,
                        'is_critical': alert.is_critical,
                        'timestamp': alert.timestamp.isoformat()
                    }
                    for alert in alerts
                ]
            
            # Save incident
            self._save_incident(incident)
            
            logger.info(f"Recorded new incident: {incident_id}")
            
            return incident_id
        
        except Exception as e:
            logger.error(f"Failed to record incident: {e}")
            return None
    
    def _save_incident(self, incident):
        """
        Save an incident to disk.
        
        Args:
            incident: Incident to save
        """
        try:
            # Create incident directory
            incident_dir = os.path.join(self.data_dir, incident.incident_id)
            os.makedirs(incident_dir, exist_ok=True)
            
            # Save frames
            frames_dir = os.path.join(incident_dir, "frames")
            os.makedirs(frames_dir, exist_ok=True)
            
            for i, frame_data in enumerate(incident.frames):
                frame_path = os.path.join(frames_dir, f"frame_{i:04d}.jpg")
                cv2.imwrite(frame_path, frame_data['frame'])
            
            # Generate hash of all files
            hash_obj = hashlib.sha256()
            
            for root, _, files in os.walk(incident_dir):
                for file in sorted(files):
                    file_path = os.path.join(root, file)
                    with open(file_path, 'rb') as f:
                        hash_obj.update(f.read())
            
            incident.hash = hash_obj.hexdigest()
            
            # Save metadata
            metadata = incident.to_dict()
            metadata_path = os.path.join(incident_dir, "metadata.json")
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Save to database
            self._save_to_database(incident)
            
            logger.info(f"Saved incident {incident.incident_id} to disk")
        
        except Exception as e:
            logger.error(f"Failed to save incident {incident.incident_id}: {e}")
    
    def _save_to_database(self, incident):
        """
        Save incident metadata to the database.
        
        Args:
            incident: Incident to save
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Convert metadata to JSON string
            metadata_json = json.dumps(incident.metadata)
            
            # Insert or replace incident
            cursor.execute('''
                INSERT OR REPLACE INTO incidents
                (id, timestamp, risk_type, risk_score, location, metadata, hash, frame_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                incident.incident_id,
                incident.timestamp.isoformat(),
                incident.risk_type,
                incident.risk_score,
                json.dumps(incident.location) if incident.location else None,
                metadata_json,
                incident.hash,
                len(incident.frames)
            ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Saved incident {incident.incident_id} to database")
        
        except Exception as e:
            logger.error(f"Failed to save incident {incident.incident_id} to database: {e}")
    
    def get_incident(self, incident_id):
        """
        Get an incident by ID.
        
        Args:
            incident_id: ID of the incident to retrieve
            
        Returns:
            Incident object or None if not found
        """
        try:
            # Check if incident directory exists
            incident_dir = os.path.join(self.data_dir, incident_id)
            
            if not os.path.exists(incident_dir):
                logger.warning(f"Incident directory not found: {incident_dir}")
                return None
            
            # Load metadata
            metadata_path = os.path.join(incident_dir, "metadata.json")
            
            if not os.path.exists(metadata_path):
                logger.warning(f"Incident metadata not found: {metadata_path}")
                return None
            
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Create incident object
            incident = Incident.from_dict(metadata)
            
            # Load frames
            frames_dir = os.path.join(incident_dir, "frames")
            
            if os.path.exists(frames_dir):
                frame_files = sorted([f for f in os.listdir(frames_dir) if f.startswith("frame_") and f.endswith(".jpg")])
                
                for frame_file in frame_files:
                    frame_path = os.path.join(frames_dir, frame_file)
                    frame = cv2.imread(frame_path)
                    
                    if frame is not None:
                        incident.add_frame(frame)
            
            return incident
        
        except Exception as e:
            logger.error(f"Failed to get incident {incident_id}: {e}")
            return None
    
    def list_incidents(self, limit=100, offset=0, sort_by="timestamp", sort_order="desc"):
        """
        List incidents.
        
        Args:
            limit: Maximum number of incidents to return
            offset: Offset for pagination
            sort_by: Field to sort by
            sort_order: "asc" or "desc"
            
        Returns:
            List of incident metadata dictionaries
        """
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Validate sort parameters
            valid_sort_fields = ["timestamp", "risk_score", "risk_type", "frame_count"]
            if sort_by not in valid_sort_fields:
                sort_by = "timestamp"
            
            valid_sort_orders = ["asc", "desc"]
            if sort_order.lower() not in valid_sort_orders:
                sort_order = "desc"
            
            # Query incidents
            cursor.execute(f'''
                SELECT * FROM incidents
                ORDER BY {sort_by} {sort_order}
                LIMIT ? OFFSET ?
            ''', (limit, offset))
            
            rows = cursor.fetchall()
            
            incidents = []
            for row in rows:
                incident_dict = dict(row)
                
                # Parse JSON fields
                if incident_dict['metadata']:
                    incident_dict['metadata'] = json.loads(incident_dict['metadata'])
                
                if incident_dict['location']:
                    incident_dict['location'] = json.loads(incident_dict['location'])
                
                incidents.append(incident_dict)
            
            conn.close()
            
            return incidents
        
        except Exception as e:
            logger.error(f"Failed to list incidents: {e}")
            return []
    
    def delete_incident(self, incident_id):
        """
        Delete an incident.
        
        Args:
            incident_id: ID of the incident to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if incident directory exists
            incident_dir = os.path.join(self.data_dir, incident_id)
            
            if os.path.exists(incident_dir):
                # Delete the directory
                shutil.rmtree(incident_dir)
            
            # Delete from database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("DELETE FROM incidents WHERE id = ?", (incident_id,))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Deleted incident: {incident_id}")
            
            return True
        
        except Exception as e:
            logger.error(f"Failed to delete incident {incident_id}: {e}")
            return False
    
    def cleanup_old_incidents(self):
        """
        Delete incidents older than the retention period.
        
        Returns:
            Number of incidents deleted
        """
        try:
            # Calculate cutoff date
            cutoff_date = datetime.now() - timedelta(days=self.retention_days)
            cutoff_str = cutoff_date.isoformat()
            
            # Query old incidents
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("SELECT id FROM incidents WHERE timestamp < ?", (cutoff_str,))
            
            old_incidents = [row['id'] for row in cursor.fetchall()]
            
            # Delete each incident
            count = 0
            for incident_id in old_incidents:
                if self.delete_incident(incident_id):
                    count += 1
            
            conn.close()
            
            logger.info(f"Cleaned up {count} old incidents")
            
            return count
        
        except Exception as e:
            logger.error(f"Failed to cleanup old incidents: {e}")
            return 0
    
    def generate_report(self, incident_id, output_path=None):
        """
        Generate a report for an incident.
        
        Args:
            incident_id: ID of the incident
            output_path: Path to save the report, or None to use a default path
            
        Returns:
            Path to the generated report or None if failed
        """
        try:
            # Get the incident
            incident = self.get_incident(incident_id)
            
            if incident is None:
                logger.warning(f"Incident not found: {incident_id}")
                return None
            
            # Create report directory
            if output_path is None:
                reports_dir = os.path.join(self.data_dir, "reports")
                os.makedirs(reports_dir, exist_ok=True)
                output_path = os.path.join(reports_dir, f"report_{incident_id}.html")
            
            # Generate HTML report
            with open(output_path, 'w') as f:
                f.write(f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <title>Incident Report: {incident_id}</title>
                    <style>
                        body {{ font-family: Arial, sans-serif; margin: 20px; }}
                        h1 {{ color: #333; }}
                        .incident-info {{ margin-bottom: 20px; }}
                        .incident-frames {{ display: flex; flex-wrap: wrap; }}
                        .frame {{ margin: 10px; border: 1px solid #ddd; padding: 10px; }}
                        .frame img {{ max-width: 400px; }}
                        .metadata {{ margin-top: 20px; padding: 10px; background-color: #f5f5f5; }}
                        .hash {{ font-family: monospace; word-break: break-all; }}
                    </style>
                </head>
                <body>
                    <h1>Incident Report</h1>
                    
                    <div class="incident-info">
                        <p><strong>Incident ID:</strong> {incident.incident_id}</p>
                        <p><strong>Timestamp:</strong> {incident.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</p>
                        <p><strong>Risk Type:</strong> {incident.risk_type}</p>
                        <p><strong>Risk Score:</strong> {incident.risk_score:.2f}</p>
                    </div>
                    
                    <h2>Alerts</h2>
                    <ul>
                """)
                
                # Add alerts
                if 'alerts' in incident.metadata:
                    for alert in incident.metadata['alerts']:
                        f.write(f"""
                        <li>
                            <strong>{alert['message']}</strong>
                            <p>{alert['explanation']}</p>
                            <p>Critical: {'Yes' if alert['is_critical'] else 'No'}</p>
                        </li>
                        """)
                else:
                    f.write("<li>No alerts recorded</li>")
                
                f.write("</ul>")
                
                # Add frames
                f.write("<h2>Frames</h2>")
                f.write("<div class='incident-frames'>")
                
                incident_dir = os.path.join(self.data_dir, incident_id)
                frames_dir = os.path.join(incident_dir, "frames")
                
                if os.path.exists(frames_dir):
                    frame_files = sorted([f for f in os.listdir(frames_dir) if f.startswith("frame_") and f.endswith(".jpg")])
                    
                    for i, frame_file in enumerate(frame_files):
                        frame_path = os.path.join("frames", frame_file)
                        f.write(f"""
                        <div class="frame">
                            <h3>Frame {i+1}</h3>
                            <img src="{frame_path}" alt="Frame {i+1}">
                        </div>
                        """)
                else:
                    f.write("<p>No frames available</p>")
                
                f.write("</div>")
                
                # Add hash
                f.write(f"""
                    <div class="metadata">
                        <h2>Integrity Information</h2>
                        <p><strong>Hash:</strong> <span class="hash">{incident.hash}</span></p>
                        <p>This hash can be used to verify the integrity of the incident data.</p>
                    </div>
                </body>
                </html>
                """)
            
            logger.info(f"Generated report for incident {incident_id}: {output_path}")
            
            return output_path
        
        except Exception as e:
            logger.error(f"Failed to generate report for incident {incident_id}: {e}")
            return None
