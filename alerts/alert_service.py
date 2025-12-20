"""
Legacy alert service - now redirects to regime alert service.
This file is kept for backward compatibility. All alerts now use the regime classifier.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import and run the regime alert service
from alerts.regime_alert_service import main

if __name__ == "__main__":
    print("Note: alert_service.py now redirects to regime_alert_service.py")
    print("=" * 60)
    main()

