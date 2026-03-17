"""Role and party definitions for Secret Hitler."""
from enum import Enum


class Party(Enum):
    LIBERAL = "Liberal"
    FASCIST = "Fascist"


class Role(Enum):
    LIBERAL = "Liberal"
    FASCIST = "Fascist"
    HITLER = "Hitler"

    @property
    def party(self) -> Party:
        if self == Role.LIBERAL:
            return Party.LIBERAL
        return Party.FASCIST


# Role composition by player count
ROLE_COUNTS = {
    5:  [Role.LIBERAL] * 3 + [Role.FASCIST] * 1 + [Role.HITLER],
    6:  [Role.LIBERAL] * 4 + [Role.FASCIST] * 1 + [Role.HITLER],
    7:  [Role.LIBERAL] * 4 + [Role.FASCIST] * 2 + [Role.HITLER],
    8:  [Role.LIBERAL] * 5 + [Role.FASCIST] * 2 + [Role.HITLER],
    9:  [Role.LIBERAL] * 5 + [Role.FASCIST] * 3 + [Role.HITLER],
    10: [Role.LIBERAL] * 6 + [Role.FASCIST] * 3 + [Role.HITLER],
}
