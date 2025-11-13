"""
Data structures for QMIX Multi-Agent Coverage system.
"""

from dataclasses import dataclass, field
from typing import Tuple, List, Dict, Optional
import networkx as nx


@dataclass
class RobotState:
    """Represents the state of a single robot."""
    position: Tuple[int, int]
    orientation: float  # in radians


@dataclass
class WorldState:
    """Represents the complete world state."""
    graph: nx.Graph  # Should only contain traversable nodes
    robots: List[RobotState]


@dataclass
class CoverageMetrics:
    """Class to track coverage performance metrics (DQN Style + QMIX)."""
    # STEP-LEVEL metrics (appended each step)
    total_coverage: List[float] = field(default_factory=list)
    coverage_rate: List[float] = field(default_factory=list)
    redundant_coverage: List[float] = field(default_factory=list)
    agent_rewards: Dict[int, List[float]] = field(default_factory=lambda: {i: [] for i in range(10)})
    qmix_loss: List[float] = field(default_factory=list)
    agent_q_values: Dict[int, List[List[float]]] = field(default_factory=lambda: {i: [] for i in range(10)})
    q_tot_values: List[float] = field(default_factory=list)
    planning_times: List[float] = field(default_factory=list)
    communication_events: List[int] = field(default_factory=list)
    execution_times: List[float] = field(default_factory=list)
    epsilon_values: Dict[int, List[float]] = field(default_factory=lambda: {i: [] for i in range(10)})
    
    # EPISODE-LEVEL metrics (appended once per episode)
    episode_coverage: List[float] = field(default_factory=list)  # Final coverage % at end of episode
    episode_steps: List[int] = field(default_factory=list)  # Number of steps taken in episode
    episode_rewards: List[float] = field(default_factory=list)  # Total reward accumulated in episode
    episode_avg_loss: List[float] = field(default_factory=list)  # Average QMIX loss during episode
    
    # Threshold tracking
    final_coverage_reached: bool = False
    steps_to_threshold: Optional[int] = None


@dataclass
class CommunicationEvent:
    """Represents a communication event between agents."""
    sender_id: int
    receiver_id: int
    data_type: str  # 'coverage', 'position', etc.
    timestamp: float


@dataclass
class Room:
    """Represents a room in the map generation."""
    r1: int
    c1: int
    height: int
    width: int

    @property
    def r2(self) -> int:
        return self.r1 + self.height

    @property
    def c2(self) -> int:
        return self.c1 + self.width

    @property
    def center(self) -> Tuple[int, int]:
        return (self.r1 + self.height // 2, self.c1 + self.width // 2)

    def intersects(self, other: 'Room') -> bool:
        return (self.r1 < other.r2 and self.r2 > other.r1 and
                self.c1 < other.c2 and self.c2 > other.c1)


# Map generation constants
NUM_ROOMS_RANGE = (6, 12)
ROOM_SIZE_RANGE = (4, 8)
