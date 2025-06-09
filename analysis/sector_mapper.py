"""
섹터 매핑 엔진
"""
import yfinance as yf

class SectorMappingEngine:
    """종목 → 섹터 매핑 (yfinance 우선 → 캐시 → 하드코딩 → AI)"""

    SECTOR_MAPPING_CACHE = {}
    KNOWN_MAPPINGS = {
        # 하드코딩 매핑들...
    }

    def __init__(self, api_manager):
        self.api_manager = api_manager

    async def get_sector(self, symbol: str) -> str:
        # 1. 캐시
        if symbol in self.SECTOR_MAPPING_CACHE:
            return self.SECTOR_MAPPING_CACHE[symbol]

        # 2. yfinance 조회 (무료)
        sector = await self._fetch_sector_yfinance(symbol)
        if sector:
            return self._cache_and_return(symbol, sector)

        # 3. 하드코딩 매핑
        if symbol in self.KNOWN_MAPPINGS:
            return self._cache_and_return(symbol, self.KNOWN_MAPPINGS[symbol])

        # 4. AI 매핑 (최후 수단)
        sector = await self._ai_sector_mapping(symbol)
        return self._cache_and_return(symbol, sector)

    def _cache_and_return(self, symbol: str, sector: str) -> str:
        self.SECTOR_MAPPING_CACHE[symbol] = sector
        return sector

    async def _fetch_sector_yfinance(self, symbol: str) -> str | None:
        try:
            tk = yf.Ticker(symbol)
            info = tk.info
            sector = info.get("sector")
            if sector:
                return sector.replace(" ", "")
        except Exception:
            pass
        return None

    async def _ai_sector_mapping(self, symbol: str) -> str:
        """AI를 이용한 섹터 매핑 (임시 구현)"""
        try:
            # 기본 섹터 반환 (향후 AI 연동)
            return "Technology"
        except:
            return "Unknown"
