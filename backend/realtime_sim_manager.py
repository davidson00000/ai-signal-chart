"""
Realtime Simulation Manager for Auto Sim Lab

This module manages multiple realtime simulation sessions.
It handles session lifecycle (start, stop, query) and coordinates
the background tick loop.
"""

import uuid
import asyncio
from datetime import datetime
from typing import Dict, Optional, Any
from contextlib import asynccontextmanager

from backend.realtime_data_source import RealtimeDataSource, get_yahoo_data_source
from backend.realtime_sim_engine import RealtimeSimEngine
from backend.auto_sim_lab import AutoSimConfig


class RealtimeSimManager:
    """
    Manager for realtime simulation sessions.
    
    Handles:
    - Creating and tracking simulation sessions
    - Starting/stopping simulations
    - Querying session state
    - Background tick coordination
    """
    
    def __init__(self, data_source: Optional[RealtimeDataSource] = None):
        """
        Initialize the simulation manager.
        
        Args:
            data_source: Data source for price feeds (defaults to Yahoo)
        """
        self.data_source = data_source or get_yahoo_data_source()
        self.sessions: Dict[str, RealtimeSimEngine] = {}
        self._tick_task: Optional[asyncio.Task] = None
        self._running = False
    
    def start_session(self, config: AutoSimConfig) -> str:
        """
        Start a new realtime simulation session.
        
        Args:
            config: Simulation configuration
            
        Returns:
            Session ID string
        """
        session_id = f"rt_{uuid.uuid4().hex[:8]}_{int(datetime.utcnow().timestamp())}"
        
        engine = RealtimeSimEngine(
            session_id=session_id,
            config=config,
            data_source=self.data_source
        )
        
        self.sessions[session_id] = engine
        print(f"[RealtimeSimManager] Started session {session_id} for {config.symbol}")
        
        return session_id
    
    def stop_session(self, session_id: str) -> bool:
        """
        Stop a running simulation session.
        
        Args:
            session_id: Session ID to stop
            
        Returns:
            True if session was stopped, False if not found
        """
        if session_id in self.sessions:
            engine = self.sessions[session_id]
            engine.stop()
            print(f"[RealtimeSimManager] Stopped session {session_id}")
            return True
        return False
    
    def remove_session(self, session_id: str) -> bool:
        """
        Remove a session from the manager.
        
        Args:
            session_id: Session ID to remove
            
        Returns:
            True if session was removed, False if not found
        """
        if session_id in self.sessions:
            del self.sessions[session_id]
            return True
        return False
    
    def get_state(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the current state of a simulation session.
        
        Args:
            session_id: Session ID to query
            
        Returns:
            Session state dict or None if not found
        """
        if session_id in self.sessions:
            return self.sessions[session_id].get_state()
        return None
    
    def list_sessions(self) -> Dict[str, Dict[str, Any]]:
        """
        List all active sessions with basic info.
        
        Returns:
            Dict mapping session_id to basic info
        """
        result = {}
        for session_id, engine in self.sessions.items():
            result[session_id] = {
                "symbol": engine.symbol,
                "timeframe": engine.timeframe,
                "is_running": engine.is_running,
                "tick_count": engine.tick_count,
                "current_equity": round(engine.equity, 2)
            }
        return result
    
    async def tick_all_sessions(self) -> int:
        """
        Process one tick for all active sessions.
        
        Returns:
            Number of sessions that were ticked
        """
        ticked = 0
        for session_id, engine in list(self.sessions.items()):
            if engine.is_running:
                try:
                    await engine.tick()
                    ticked += 1
                except Exception as e:
                    print(f"[RealtimeSimManager] Error ticking {session_id}: {e}")
        return ticked
    
    async def start_background_loop(self, interval_seconds: float = 10.0):
        """
        Start the background tick loop.
        
        Args:
            interval_seconds: Seconds between ticks
        """
        self._running = True
        print(f"[RealtimeSimManager] Starting background loop (interval: {interval_seconds}s)")
        
        while self._running:
            try:
                active_count = sum(1 for e in self.sessions.values() if e.is_running)
                if active_count > 0:
                    await self.tick_all_sessions()
                    print(f"[RealtimeSimManager] Ticked {active_count} active sessions")
            except Exception as e:
                print(f"[RealtimeSimManager] Loop error: {e}")
            
            await asyncio.sleep(interval_seconds)
    
    def stop_background_loop(self):
        """Stop the background tick loop."""
        self._running = False
        print("[RealtimeSimManager] Background loop stopped")


# Global singleton instance
_realtime_manager: Optional[RealtimeSimManager] = None


def get_realtime_manager() -> RealtimeSimManager:
    """Get or create the global RealtimeSimManager instance."""
    global _realtime_manager
    if _realtime_manager is None:
        _realtime_manager = RealtimeSimManager()
    return _realtime_manager
