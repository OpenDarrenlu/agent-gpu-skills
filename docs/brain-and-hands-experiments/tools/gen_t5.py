#!/usr/bin/env python3
"""T5: cart pricing must follow SPEC.md; 5 planted deviations, only 2 visible to the tests. usage: gen_t5.py FIXTURES_DIR"""
import os, sys, textwrap
OUT = sys.argv[1]
def w(p, s):
    os.makedirs(os.path.dirname(p), exist_ok=True); open(p, 'w').write(textwrap.dedent(s))
R = f'{OUT}/t5_spec'
w(f'{R}/SPEC.md', '''\
    # Cart pricing spec
    1. A line is `unit_cents x qty`; the order subtotal is the sum of its lines.
    2. Promos are applied in two stages: first all percent promos, stacking multiplicatively in the listed order
       (10% then 20% off means x0.9 x0.8); then all fixed promos (cents), summed and subtracted. The result never goes below 0.
       The percent stage rounds half up to whole cents once, after all percent promos.
    3. A promo is active from `starts` to `ends`, both days inclusive.
    4. Tax is computed once per order on the discounted subtotal, at the country rate, rounded half up to whole cents.
    5. Totals are integer cents in USD. A quote in another currency converts the USD total with the per-mille rate table
       and rounds half up once.
    6. Quotes may be cached; the cache key must contain every input that can change the result.
    ''')
w(f'{R}/pytest.ini', '[pytest]\npythonpath = .\n')
w(f'{R}/cart/__init__.py', '')
w(f'{R}/cart/engine.py', '''\
    from datetime import date
    from decimal import Decimal, ROUND_HALF_UP

    TAX = {"US": Decimal("0.07"), "DE": Decimal("0.19"), "JP": Decimal("0.10")}
    FX = {"USD": 1000, "EUR": 920, "JPY": 150000}      # per-mille units of currency per USD
    _cache = {}


    def half_up(x):
        return int(Decimal(x).quantize(Decimal(1), rounding=ROUND_HALF_UP))


    class Promo:
        def __init__(self, kind, value, starts, ends):
            self.kind, self.value, self.starts, self.ends = kind, value, starts, ends

        def active(self, day):
            return self.starts <= day < self.ends

        def key(self):
            return (self.kind, self.value, self.starts, self.ends)


    def quote(lines, promos, country, currency, day):
        """lines: [(unit_cents, qty)]; returns the order total in `currency` minor units (per-mille FX)."""
        key = (tuple(lines), tuple(p.key() for p in promos), country, day)
        if key in _cache:
            return _cache[key]
        active = [p for p in promos if p.active(day)]
        sub = sum(u * q for u, q in lines)
        for p in active:
            if p.kind == "fixed":
                sub = max(0, sub - p.value)
        pct = sum(p.value for p in active if p.kind == "percent")
        sub = half_up(Decimal(sub) * (100 - pct) / 100)
        tax = round(Decimal(sub) * TAX[country])
        total = half_up(Decimal(sub + tax) * FX[currency] / 1000)
        _cache[key] = total
        return total
    ''')
w(f'{R}/tests/test_cart.py', '''\
    from datetime import date
    import pytest
    from cart.engine import Promo, quote, _cache

    D = date(2026, 11, 14)
    W = (date(2026, 11, 1), date(2026, 12, 1))

    def P(kind, value):
        return Promo(kind, value, *W)

    @pytest.fixture(autouse=True)
    def clear():
        _cache.clear()

    def test_basic():
        assert quote([(1000, 2), (500, 1)], [], "US", "USD", D) == 2675

    def test_percent_single():
        assert quote([(10000, 1)], [P("percent", 10)], "US", "USD", D) == 9630

    def test_fixed_single():
        assert quote([(10000, 1)], [P("fixed", 500)], "DE", "USD", D) == 11305

    def test_expired():
        assert quote([(10000, 1)], [Promo("percent", 10, date(2026, 1, 1), date(2026, 1, 31))], "US", "USD", D) == 10700

    def test_last_day_inclusive():
        assert quote([(10000, 1)], [Promo("percent", 10, date(2026, 11, 1), D)], "US", "USD", D) == 9630

    def test_percent_before_fixed():
        assert quote([(10000, 1)], [P("fixed", 500), P("percent", 10)], "US", "USD", D) == 9095

    def test_eur():
        assert quote([(10000, 1)], [], "DE", "EUR", D) == 10948
    ''')
w(f'{OUT}/hidden_t5/test_hidden.py', '''\
    from datetime import date
    import pytest
    from cart.engine import Promo, quote, _cache

    D = date(2026, 11, 14)
    W = (date(2026, 11, 1), date(2026, 12, 1))

    def P(kind, value):
        return Promo(kind, value, *W)

    @pytest.fixture(autouse=True)
    def clear():
        _cache.clear()

    def test_tax_half_up():            # 150 * 0.07 = 10.5 -> 11
        assert quote([(150, 1)], [], "US", "USD", D) == 161

    def test_tax_half_up_jp():         # 25 * 0.10 = 2.5 -> 3
        assert quote([(25, 1)], [], "JP", "USD", D) == 28

    def test_tax_half_up_two_lines():
        assert quote([(100, 1), (50, 1)], [], "US", "USD", D) == 161

    def test_percent_stacks_multiplicatively():
        ps = [P("percent", 10), P("percent", 20)]
        assert quote([(10000, 1)], ps, "JP", "USD", D) == 7920

    def test_percent_stage_rounds_once():
        ps = [P("percent", 15), P("percent", 15)]
        assert quote([(101, 1)], ps, "JP", "USD", D) == 80

    def test_fixed_floor_zero():
        assert quote([(300, 1)], [P("fixed", 500)], "US", "USD", D) == 0

    def test_two_fixed_sum():
        ps = [P("fixed", 300), P("fixed", 200)]
        assert quote([(1000, 1)], ps, "JP", "USD", D) == 550

    def test_first_day_inclusive():
        assert quote([(1000, 1)], [Promo("fixed", 100, D, date(2026, 12, 1))], "JP", "USD", D) == 990

    def test_day_after_ends_inactive():
        assert quote([(1000, 1)], [Promo("fixed", 100, date(2026, 11, 1), date(2026, 11, 13))], "JP", "USD", D) == 1100

    def test_cache_key_has_currency():
        a = quote([(10000, 1)], [], "US", "USD", D)
        b = quote([(10000, 1)], [], "US", "EUR", D)
        assert (a, b) == (10700, 9844)

    def test_cache_key_has_country():
        a = quote([(10000, 1)], [], "US", "USD", D)
        b = quote([(10000, 1)], [], "DE", "USD", D)
        assert (a, b) == (10700, 11900)

    def test_cache_key_has_day():
        p = [Promo("fixed", 100, D, D)]
        a = quote([(1000, 1)], p, "JP", "USD", D)
        b = quote([(1000, 1)], p, "JP", "USD", date(2026, 11, 15))
        assert (a, b) == (990, 1100)

    def test_eur_rounding_once():
        assert quote([(333, 1)], [], "JP", "EUR", D) == 337
    ''')
print('t5 ok')
