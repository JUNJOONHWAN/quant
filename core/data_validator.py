"""
데이터 검증 매니저
"""

class DataValidationManager:
    """전체 시스템 데이터 검증 및 결손 처리 매니저"""
    
    def __init__(self):
        self.strict_mode = True  # True면 결손 시 무조건 HOLD
        self.validation_rules = self._init_validation_rules()
        
    def _init_validation_rules(self) -> dict:
        """데이터 검증 규칙 정의"""
        return {
            'v_score': {
                'required_fields': ['per', 'revenue_growth', 'profit_margin', 'debt_ratio'],
                'valid_ranges': {
                    'per': (1, 200),
                    'revenue_growth': (-50, 100),
                    'profit_margin': (-20, 80),
                    'debt_ratio': (0, 300)
                }
            },
            't_score': {
                'required_fields': ['ma_above_count', 'rsi', 'macd_signal', 'volume_sigma'],
                'valid_ranges': {
                    'ma_above_count': (0, 6),
                    'rsi': (0, 100),
                    'volume_sigma': (0, 10)
                }
            },
            'f_score': {
                'required_fields': ['institutional_ownership', 'analyst_coverage', 'trading_activity'],
                'valid_ranges': {
                    'institutional_ownership': (0, 100),
                    'analyst_coverage': (0, 50),
                    'trading_activity': (0, 10)
                }
            },
            'n_score': {
                'required_fields': ['sentiment_index', 'sentiment_change_rate'],
                'valid_ranges': {
                    'sentiment_index': (0, 10),
                    'sentiment_change_rate': (-100, 100)
                }
            }
        }
    
    def validate_component_data(self, component: str, data: dict) -> dict:
        """개별 컴포넌트 데이터 검증"""
        validation_result = {
            'is_valid': True,
            'missing_fields': [],
            'invalid_values': [],
            'error_details': []
        }
        
        if component not in self.validation_rules:
            validation_result['is_valid'] = False
            validation_result['error_details'].append(f'Unknown component: {component}')
            return validation_result
            
        rules = self.validation_rules[component]
        
        # 필수 필드 검증
        for field in rules['required_fields']:
            if field not in data or data[field] is None:
                validation_result['missing_fields'].append(field)
                validation_result['is_valid'] = False
                
        # 값 범위 검증
        for field, (min_val, max_val) in rules.get('valid_ranges', {}).items():
            if field in data and data[field] is not None:
                try:
                    value = float(data[field])
                    if not (min_val <= value <= max_val):
                        validation_result['invalid_values'].append(
                            f'{field}: {value} (범위: {min_val}-{max_val})'
                        )
                        validation_result['is_valid'] = False
                except (ValueError, TypeError):
                    validation_result['invalid_values'].append(
                        f'{field}: 숫자 변환 불가'
                    )
                    validation_result['is_valid'] = False
                    
        return validation_result
