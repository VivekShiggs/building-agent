"""Address resolution — multi-tier strategy for house numbers and street names.

Tier 1: OSM addr:* tags (fast, free, no rate limit)
Tier 2: Nominatim reverse geocode (rate-limited, 1 req/s)
Tier 3: Log gaps for manual resolution

Security:
  - Rate limiting to respect Nominatim usage policy
  - Request timeouts to prevent hanging
  - User-agent set to identify the application
  - Only HTTPS URLs used
"""

import logging
import time
from typing import Any, Dict, Optional

import requests

logger = logging.getLogger(__name__)

NOMINATIM_DEFAULT_URL = "https://nominatim.openstreetmap.org"
USER_AGENT = "BuildingAgent/1.0 (data-collection-agent)"
REQUEST_TIMEOUT = 15


class AddressResolver:
    """Resolve street addresses for building centroids.

    Args:
        nominatim_url: Nominatim endpoint URL
        rate_delay: Seconds between Nominatim requests
        max_retries: Max retries on failure
    """

    def __init__(
        self,
        nominatim_url: str = NOMINATIM_DEFAULT_URL,
        rate_delay: float = 1.0,
        max_retries: int = 3,
    ):
        self._nominatim_url = nominatim_url.rstrip("/")
        self._rate_delay = rate_delay
        self._max_retries = max_retries
        self._last_request_time = 0.0

    def resolve(self, lat: float, lon: float) -> Dict[str, Any]:
        """Resolve address for a (lat, lon) coordinate pair.

        Currently uses Nominatim reverse geocode (Tier 2).
        Tier 1 (OSM addr:* on building footprints) is handled during OSM data fetch.

        Returns:
            dict with keys: house_number, street, city, postcode, status
        """
        result: Dict[str, Any] = {
            "house_number": None,
            "street": None,
            "city": None,
            "postcode": None,
            "status": "not_found",
        }

        try:
            addr = self._nominatim_reverse(lat, lon)
            if addr:
                result.update(addr)
                result["status"] = "geocoded"
        except Exception as e:
            logger.debug("Nominatim failed for (%.5f, %.5f): %s", lat, lon, e)
            result["status"] = "error"

        return result

    def _nominatim_reverse(self, lat: float, lon: float) -> Optional[Dict[str, Any]]:
        """Reverse geocode via Nominatim API with rate limiting."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._rate_delay:
            time.sleep(self._rate_delay - elapsed)

        url = f"{self._nominatim_url}/reverse"
        params = {
            "lat": lat,
            "lon": lon,
            "format": "jsonv2",
            "addressdetails": 1,
            "zoom": 18,
        }
        headers = {"User-Agent": USER_AGENT}

        for attempt in range(self._max_retries):
            try:
                resp = requests.get(
                    url,
                    params=params,
                    headers=headers,
                    timeout=REQUEST_TIMEOUT,
                )
                self._last_request_time = time.time()

                if resp.status_code == 429:
                    wait = (attempt + 1) * 2.0
                    logger.warning("Nominatim rate limited, waiting %.0fs", wait)
                    time.sleep(wait)
                    continue

                resp.raise_for_status()
                data = resp.json()

                if "address" not in data or not data["address"]:
                    return None

                addr = data["address"]
                return {
                    "house_number": addr.get("house_number"),
                    "street": (
                        addr.get("road")
                        or addr.get("pedestrian")
                        or addr.get("street")
                    ),
                    "city": (
                        addr.get("city")
                        or addr.get("town")
                        or addr.get("village")
                        or addr.get("municipality")
                    ),
                    "postcode": addr.get("postcode"),
                }

            except requests.RequestException as e:
                logger.debug(
                    "Nominatim attempt %d/%d failed: %s",
                    attempt + 1,
                    self._max_retries,
                    e,
                )
                if attempt < self._max_retries - 1:
                    time.sleep((attempt + 1) * 1.0)

        return None
