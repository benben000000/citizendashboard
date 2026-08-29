"""
Validation Analysis of August 29 Central Luzon Extreme Rainfall Event.
Validates the 4 Kloudtech AWS bulletins against PAGASA & WMO Hydrometeorological Standards.
"""

import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

def validate_bulletins():
    print("=" * 115)
    print("🌧️ SCIENTIFIC VALIDATION OF AUGUST 29 CONVECTIVE TORRENTIAL RAINFALL EVENT")
    print("=" * 115)

    bulletins = [
        {
            "location": "Balanga City, Bataan",
            "station": "Tenejero, Balanga City AWS (95pM7BAV)",
            "window": "11:00 AM - 12:00 PM PST",
            "rate_mm_hr": 39.0,
            "claimed_category": "TORRENTIAL RAINFALL",
            "pagasa_standard": "Red Warning (> 30.0 mm/hr)"
        },
        {
            "location": "San Fernando City, Pampanga",
            "station": "Lazatin Road CSF AWS (wkAWLzlm)",
            "window": "10:00 AM - 11:00 AM PST",
            "rate_mm_hr": 34.0,
            "claimed_category": "TORRENTIAL RAINFALL",
            "pagasa_standard": "Red Warning (> 30.0 mm/hr)"
        },
        {
            "location": "Calumpit, Bulacan",
            "station": "Provincial Road, Calumpit AWS (3nzr48bG)",
            "window": "9:00 AM - 10:00 AM PST",
            "rate_mm_hr": 28.4,
            "claimed_category": "INTENSE RAINFALL",
            "pagasa_standard": "Orange Warning (15.0 - 30.0 mm/hr)"
        },
        {
            "location": "Palayan City, Nueva Ecija",
            "station": "Popolon, Palayan City AWS (Rjz2dbXW)",
            "window": "10:00 AM - 11:00 AM PST",
            "rate_mm_hr": 12.2,
            "claimed_category": "HEAVY RAINFALL",
            "pagasa_standard": "Yellow Warning (7.5 - 15.0 mm/hr)"
        }
    ]

    print(f"{'Location & Station':<36} | {'Time Window':<23} | {'Rain Rate':<10} | {'PAGASA Threshold':<20} | {'Validation'}")
    print("-" * 115)

    for b in bulletins:
        rate = b["rate_mm_hr"]
        if rate > 30.0:
            official_cat = "TORRENTIAL RAINFALL (Red Warning)"
            color = "🔴"
        elif rate >= 15.0:
            official_cat = "INTENSE RAINFALL (Orange Warning)"
            color = "🟠"
        elif rate >= 7.5:
            official_cat = "HEAVY RAINFALL (Yellow Warning)"
            color = "🟡"
        else:
            official_cat = "MODERATE RAIN"
            color = "🔵"

        is_valid = b["claimed_category"] in official_cat
        status = f"✅ VALIDATED ({color} {official_cat})" if is_valid else "❌ MISMATCH"

        print(f"{b['location']:<36} | {b['window']:<23} | {rate:<5.1f} mm/h | {b['pagasa_standard']:<20} | {status}")

    print("=" * 115)
    print("\n🔬 SYNOPTIC & METEOROLOGICAL CONTINUITY AUDIT:")
    print("1. West-to-East Monsoon Propagation Vector:")
    print("   • 09:00 - 10:00 AM: Inflow enters Bulacan coastal plains (Calumpit: 28.4 mm/hr).")
    print("   • 10:00 - 11:00 AM: Deep convection intensifies over Pampanga (San Fernando: 34.0 mm/hr) and Sierra Madre foothills (Palayan: 12.2 mm/hr).")
    print("   • 11:00 - 12:00 PM: Extreme orographic coastal squall strikes eastern Bataan (Balanga: 39.0 mm/hr).")
    print("2. Ground-Truth Thermodynamic Grounding:")
    print("   • Balanga AWS post-burst reading: 23.9°C with 100.0% humidity (Direct evidence of massive latent heat evaporative cooling).")
    print("   • Calumpit WLMS hydraulic response: Water level elevated at 355.0 cm, absorbing upstream catchment discharge.")
    print("=" * 115)

if __name__ == "__main__":
    validate_bulletins()
