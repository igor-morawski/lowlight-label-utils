CATEGORIES=["person", "bicycle", "car"]
CAT_NAME_TO_ID={n:i+1 for i, n in enumerate(CATEGORIES)}
ID_TO_CAT_NAME=[None] + CATEGORIES