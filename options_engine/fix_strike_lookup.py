# Patch backtester.py to use nearest available strike
import re

code = open('backtester.py').read()

old = '''    def find_strike(right, strike):
        mask = (chain['right'] == right) & (chain['strike'].round(0) == round(strike))
        rows = chain[mask]
        return rows.iloc[0] if not rows.empty else None'''

new = '''    def find_strike(right, strike):
        sub = chain[chain['right'] == right]
        if sub.empty:
            return None
        nearest = sub.iloc[(sub['strike'] - strike).abs().argsort()[:1]]
        if abs(float(nearest['strike'].iloc[0]) - strike) > 50:
            return None  # too far
        return nearest.iloc[0]'''

code = code.replace(old, new)
open('backtester.py', 'w').write(code)
print('Patched')
