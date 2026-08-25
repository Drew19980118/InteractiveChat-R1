"""Minimal unused airport catalogue required by old outlines imports."""

# outlines/types/airports.py only iterates this value while defining its
# optional airport-code Enum. IGPO never requests that Enum.
AIRPORT_LIST = []