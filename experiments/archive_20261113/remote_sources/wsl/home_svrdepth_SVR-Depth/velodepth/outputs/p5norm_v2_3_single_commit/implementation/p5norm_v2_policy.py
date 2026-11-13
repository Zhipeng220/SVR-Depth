"""Feature-only policies. B is deliberately unavailable."""
def select(strategy, s0, s1):
 if strategy not in ('A','C'): raise ValueError('only A/C approved')
 if len(s0)!=4 or len(s1)!=4: raise ValueError('expected four feature levels')
 return s0 if strategy=='A' else s1
