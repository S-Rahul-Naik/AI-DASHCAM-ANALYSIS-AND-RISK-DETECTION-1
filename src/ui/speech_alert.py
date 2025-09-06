"""
Provides text-to-speech capability for dashcam alerts with optimized performance
"""

import logging
import threading
import time
import platform
import subprocess
import queue
import concurrent.futures
import os

logger = logging.getLogger(__name__)

class SpeechAlertSystem:
    """
    Handles text-to-speech alerts for the dashcam with minimal latency
    """
    
    def __init__(self):
        """Initialize the speech alert system with performance optimizations."""
        self.active = True
        self.alert_queue = queue.Queue()  # Thread-safe queue for alerts
        self.last_alert_time = 0
        self.last_alert_type = None
        self.alert_cooldown = 3  # Seconds between spoken alerts
        self.critical_cooldown = 1.5  # Shorter cooldown for critical alerts
        self.alert_type_cooldown = 8  # Longer cooldown between alerts of the same type
        
        # Create a thread pool for handling speech synthesis
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        
        # Create an event to signal when speech is completed
        self.speech_completed = threading.Event()
        self.speech_completed.set()  # Initially set to true (no speech in progress)
        
        # Create a cache for pre-synthesized common alerts
        self.speech_cache = {}
        
        # Start alert worker thread
        self.alert_thread = threading.Thread(target=self._alert_worker, daemon=True)
        self.alert_thread.start()
        
        # Pre-cache common alert messages in the background
        self.executor.submit(self._precache_common_alerts)
        
        logger.info("Enhanced speech alert system initialized")
    
    def _precache_common_alerts(self):
        """Pre-cache common alert messages to reduce latency."""
        common_alerts = [
            "BRAKE NOW! Vehicle ahead",
            "Caution: Vehicle ahead", 
            "Car changing to left lane",
            "Car changing to right lane",
            "CAUTION: Large pothole ahead",
            "Pothole detected",
            "Multiple potholes ahead",
            "CAUTION: Road crack ahead"
        ]
        
        # Only use caching on Windows where we can use the more efficient method
        if platform.system() == 'Windows':
            for msg in common_alerts:
                try:
                    self._prepare_windows_speech(msg)
                    time.sleep(0.1)  # Small delay to avoid system overload
                except Exception as e:
                    logger.error(f"Error pre-caching alert: {e}")
    
    def _prepare_windows_speech(self, text):
        """
        Prepare Windows speech synthesis with lower latency approach.
        Uses a PowerShell script to pre-load speech synthesis for faster execution.
        """
        # Create a unique identifier for this message
        cache_key = text.lower().replace(" ", "_")[:20]
        
        if cache_key in self.speech_cache:
            return cache_key
            
        # PowerShell script that creates a speech instance ready to speak
        ps_script = f"""
Add-Type -AssemblyName System.Speech
$voice = New-Object System.Speech.Synthesis.SpeechSynthesizer
$voice.Rate = 1  # Slightly faster than default (0)
$voice.Volume = 100
$voice.SelectVoiceByHints("Female")  # Select female voice for better clarity
$text = "{text}"
"""
        # Store the prepared script in the cache
        self.speech_cache[cache_key] = ps_script
        return cache_key
    
    def speak(self, text):
        """
        Speak text using the system's text-to-speech capability with reduced latency.
        
        Args:
            text: The text to speak
        """
        # If speech is already in progress, don't interrupt it (unless this is critical)
        if not self.speech_completed.is_set() and not text.startswith("BRAKE") and not text.startswith("CAUTION"):
            return
            
        # Reset the speech completed event
        self.speech_completed.clear()
        
        # Submit speech task to thread pool
        future = self.executor.submit(self._do_speak, text)
        # Add callback to set the event when speech is completed
        future.add_done_callback(lambda f: self.speech_completed.set())
    
    def _do_speak(self, text):
        """Actual speech implementation with platform-specific optimizations."""
        try:
            if platform.system() == 'Windows':
                # Check if this message is in the cache
                cache_key = text.lower().replace(" ", "_")[:20]
                if cache_key in self.speech_cache:
                    # Use the cached script with appended Speak command
                    ps_script = self.speech_cache[cache_key] + "\n$voice.Speak($text)"
                    # Create a temporary script file
                    script_path = os.path.join(os.environ.get('TEMP', '.'), f"speech_{cache_key}.ps1")
                    with open(script_path, 'w') as f:
                        f.write(ps_script)
                    
                    # Execute the script file (faster than passing the script as a command)
                    subprocess.run(['powershell', '-ExecutionPolicy', 'Bypass', '-File', script_path], 
                                  stdout=subprocess.DEVNULL,
                                  stderr=subprocess.DEVNULL)
                    
                    # Clean up the temporary file
                    try:
                        os.remove(script_path)
                    except:
                        pass
                else:
                    # Use direct method for uncached messages
                    subprocess.run(['powershell', '-command', 
                                   'Add-Type -AssemblyName System.Speech; ' +
                                   '$speak = New-Object System.Speech.Synthesis.SpeechSynthesizer; ' +
                                   f'$speak.Rate = 1; $speak.Speak("{text}")'], 
                                   stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL)
            
            elif platform.system() == 'Darwin':  # macOS
                # Use Mac's say command with voice and rate options for better performance
                subprocess.run(['say', '-r', '210', text],  # Slightly faster rate
                               stdout=subprocess.DEVNULL, 
                               stderr=subprocess.DEVNULL)
            
            else:  # Linux
                # Try to use espeak with optimized parameters
                try:
                    # Faster speech rate and better volume
                    subprocess.run(['espeak', '-s', '155', '-a', '150', text], 
                                  stdout=subprocess.DEVNULL, 
                                  stderr=subprocess.DEVNULL)
                except FileNotFoundError:
                    logger.warning("espeak not found, speech alerts disabled")
                    return
        except Exception as e:
            logger.error(f"Error using text-to-speech: {e}")
    
    def add_alert(self, alert):
        """
        Add an alert to be spoken with priority handling.
        
        Args:
            alert: The Alert object to speak
        """
        # Only add critical or warning alerts to the queue
        if alert.is_critical or alert.is_warning:
            # Use higher priority for critical alerts
            priority = 0 if alert.is_critical else 1
            self.alert_queue.put((priority, alert))
    
    def _alert_worker(self):
        """Background thread to process and speak alerts with improved latency."""
        while self.active:
            current_time = time.time()
            
            if self.alert_queue.empty():
                # No alerts to process
                time.sleep(0.05)  # Shorter sleep time for better responsiveness
                continue
            
            try:
                # Get the highest priority alert (lowest priority number)
                priority, alert = self.alert_queue.get_nowait()
                
                # Determine applicable cooldown based on alert type and criticality
                applicable_cooldown = self.critical_cooldown if alert.is_critical else self.alert_cooldown
                
                # Check if enough time has passed since the last alert of any type
                if current_time - self.last_alert_time >= applicable_cooldown:
                    # Get risk type from alert
                    risk_type = alert.risk.risk_type if hasattr(alert.risk, 'risk_type') else 'unknown'
                    
                    # Check if this is the same type of alert as the previous one
                    if self.last_alert_type == risk_type and \
                       current_time - self.last_alert_time < self.alert_type_cooldown:
                        # Skip this alert if we've recently announced this type
                        continue
                    
                    # Prepare speech text - shorter and more direct
                    if alert.is_critical:
                        # Critical alerts should be very direct and actionable
                        speech_text = f"{alert.message}"
                    else:
                        # Warning alerts should be informative
                        speech_text = f"{alert.message}"
                    
                    # Speak the alert with pre-caching when possible
                    self.speak(speech_text)
                    self.last_alert_time = time.time()
                    self.last_alert_type = risk_type
            
            except queue.Empty:
                pass
            
            # Sleep briefly to avoid consuming CPU
            time.sleep(0.05)  # Shorter sleep time for better responsiveness
    
    def shutdown(self):
        """Shutdown the speech alert system."""
        self.active = False
        if self.alert_thread.is_alive():
            self.alert_thread.join(timeout=1.0)
        logger.info("Speech alert system shut down")
