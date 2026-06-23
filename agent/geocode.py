"""Slovak city geocoding — hardcoded bounding boxes for major cities.

No external API calls needed. All cities are predefined with approximate
bounding boxes covering the built-up area of the city.
"""

from typing import Optional


# [west, south, east, north] in WGS84 (EPSG:4326)
SLOVAK_CITIES: dict[str, list[float]] = {
    "Bratislava": [16.85, 48.10, 17.22, 48.25],
    "Trnava": [17.52, 48.34, 17.64, 48.41],
    "Nitra": [18.05, 48.28, 18.14, 48.34],
    "Trenčín": [18.02, 48.86, 18.12, 48.93],
    "Žilina": [18.70, 49.19, 18.80, 49.26],
    "Banská Bystrica": [19.10, 48.69, 19.20, 48.78],
    "Prešov": [21.20, 48.95, 21.30, 49.04],
    "Košice": [21.20, 48.67, 21.30, 48.76],
    "Poprad": [20.26, 49.02, 20.35, 49.09],
    "Martin": [18.89, 49.04, 18.98, 49.11],
    "Prievidza": [18.61, 48.74, 18.70, 48.81],
    "Zvolen": [19.10, 48.54, 19.19, 48.61],
    "Komárno": [18.07, 47.74, 18.20, 47.81],
    "Dunajská Streda": [17.35, 47.97, 17.44, 48.04],
    "Senec": [17.36, 48.20, 17.44, 48.25],
    "Pezinok": [17.24, 48.27, 17.32, 48.32],
    "Lučenec": [19.80, 48.31, 19.89, 48.37],
    "Liptovský Mikuláš": [19.58, 49.06, 19.68, 49.13],
    "Michalovce": [21.89, 48.73, 21.97, 48.78],
    "Piešťany": [17.80, 48.57, 17.86, 48.61],
    "Topoľčany": [18.15, 48.55, 18.20, 48.58],
    "Humenné": [21.90, 48.92, 21.97, 48.96],
    "Bardejov": [21.28, 49.27, 21.34, 49.31],
    "Rožňava": [20.51, 48.64, 20.57, 48.68],
    "Nové Zámky": [18.14, 47.96, 18.20, 48.00],
    "Levice": [18.58, 48.19, 18.63, 48.24],
}


def list_cities() -> list[str]:
    """Return list of available Slovak cities, sorted alphabetically."""
    return sorted(SLOVAK_CITIES.keys())


def lookup_city(name: str) -> Optional[list[float]]:
    """Get bounding box [west, south, east, north] for a Slovak city.

    Args:
        name: City name (case-insensitive)

    Returns:
        Bounding box or None if not found
    """
    if name in SLOVAK_CITIES:
        return SLOVAK_CITIES[name]

    name_lower = name.lower().strip()
    for city, bbox in SLOVAK_CITIES.items():
        if city.lower() == name_lower:
            return bbox

    return None
