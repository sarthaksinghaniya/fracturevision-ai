import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.evaluate import main


if __name__ == "__main__":
    main()
