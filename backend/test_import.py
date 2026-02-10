import sys
print(sys.executable)
try:
    import joblib
    print(f"Joblib imported: {joblib.__version__}")
except ImportError as e:
    print(f"Error: {e}")
try:
    import numpy
    print("Numpy imported")
except ImportError as e:
    print(f"Error numpy: {e}")
try:
    import pandas
    print("Pandas imported")
except ImportError as e:
    print(f"Error pandas: {e}")
