export const WEATHER_REFERENCES = {
    "Temperature": [
    {
      term: "Cool",
      threshold: "0-24 °C",
      color: "#00BFFF",
      definition:
        "Comfortable to slightly chilly temperature; light clothing may be needed. Suitable for most outdoor activities."
    },
    {
      term: "Moderate",
      threshold: "25-29 °C",
      color: "#7FFF00",
      definition:
        "Pleasant warm temperature; light clothing recommended. Comfortable for outdoor activities."
    },
    {
      term: "Warm",
      threshold: "30-34 °C",
      color: "#FFBC01",
      definition:
        "Warm temperature; prolonged exposure may cause discomfort. Stay hydrated and avoid strenuous activity during peak hours."
    },
    {
      term: "Hot",
      threshold: "35-39 °C",
      color: "#FF9000",
      definition:
        "High temperature; risk of heat stress increases with prolonged outdoor activity. Drink plenty of water and seek shade if possible."
    },
    {
      term: "Extreme Heat",
      threshold: "40 °C+",
      color: "#E63946",
      definition:
        "Very high temperature; dangerous for outdoor activity. Risk of heat stroke; avoid outdoor exposure and stay in cool or air-conditioned areas."
    },
  ],
  "Heat Index": [
    {
      term: "Caution",
      threshold: "27-32 °C",
      color: "#FBF300",
      definition:
        "Fatigue is possible with prolonged exposure and activity. Continuing activity could lead to heat cramps.",
    },
    {
      term: "Extreme Caution",
      threshold: "33-41 °C",
      color: "#F6CC47",
      definition:
        "Heat cramps and heat exhaustion are possible. continuing activity could lead to heat stroke.",
    },
    {
      term: "Danger",
      threshold: "42-51 °C",
      color: "#EC6F31",
      definition:
        "Heat cramps and heat exhaustion are likely; heat stroke is probable with continued exposure.",
    },
    {
      term: "Extreme Danger",
      threshold: "52 °C+",
      color: "#C13030",
      definition: "Heat stroke is imminent.",
    },
  ],
  "Wind Speed": [
    // {
    //   term: "Light Winds",
    //   color: "var(--foreground)",
    //   threshold: "19 KPH",
    //   definition:
    //     "Gentle breeze; leaves rustle, smoke rises vertically; generally safe and comfortable conditions.",
    // },
    // {
    //   term: "Moderate Winds",
    //   color: "var(--foreground)",
    //   threshold: "20-29 KPH",
    //   definition: "Noticeable breeze; leaves and small branches move; outdoor activities continue without difficulty.",
    // },
    // {
    //   term: "Strong Winds",
    //   color: "var(--foreground)",
    //   threshold: "30-38 KPH",
    //   definition:
    //     "Strong breeze; small branches in motion, dust and loose paper raised; umbrellas difficult to use.",
    // },
    {
      term: "Tropical Depression - Level Winds",
      color: "#00CCFF",
      subtitle: "Wind Signal 1",
      threshold: "39-61 KPH",
      definition: `Wind strong enough to sway large branches; small trees in motion; light structural damage possible; caution advised.`,
      leadTime: "36 Hours",
      subcategory: "cyclone",
    },
    {
      term: "Tropical Storm - Level Winds",
      color: "#FBF300",
      subtitle: "Wind Signal 2",
      threshold: "62-88 KPH",
      definition: `Considerable wind; trees may be uprooted; minor structural damage; hazardous to outdoor activities.`,
      leadTime: "24 Hours",
      subcategory: "cyclone",
    },
    {
      term: "Severe Tropical Storm - Level Winds",
      color: "#FFA800",
      subtitle: "Wind Signal 3",
      threshold: "89-117 KPH",
      definition: `Heavy wind; widespread damage to weak structures; large branches broken; danger to light outdoor objects.`,
      leadTime: "18 Hours",
      subcategory: "cyclone",
    },
    {
      term: "Typhoon - Level Winds",
      color: "#E63946",
      subtitle: "Wind Signal 4",
      threshold: "118-184 KPH",
      definition: `Violent wind; uprooted trees; significant structural damage; dangerous to all outdoor activities; stay indoors.`,
      leadTime: "18 Hours",
      subcategory: "cyclone",
    },
    {
      term: "Super Typhoon - Level Winds",
      color: "#CB00CE",
      subtitle: "Wind Signal 5",
      threshold: "185 KPH+",
      definition: `Extremely violent winds; widespread devastation; catastrophic damage expected; emergency measures imperative.`,
      leadTime: "12 Hours",
      subcategory: "cyclone",
    },

  ],
  // "Cyclone": [
  //   {
  //     term: "Tropical Depression - Level Winds",
  //     color: "#D94835",
  //     subtitle: "Wind Signal 1",
  //     threshold: "39-61KPH",
  //     definition: `Wind strong enough to sway large branches; small trees in motion; light structural damage possible; caution advised.`,
  //     leadTime: "36 Hours",
  //     subcategory: "cyclone",
  //   },
  //   {
  //     term: "Tropical Storm - Level Winds",
  //     color: "#FBF300",
  //     subtitle: "Wind Signal 2",
  //     threshold: "62-88KPH",
  //     definition: `Considerable wind; trees may be uprooted; minor structural damage; hazardous to outdoor activities.`,
  //     leadTime: "24 Hours",
  //     subcategory: "cyclone",
  //   },
  //   {
  //     term: "Severe Tropical Storm - Level Winds",
  //     color: "#FFA800",
  //     subtitle: "Wind Signal 3",
  //     threshold: "89-117KPH",
  //     definition: `Heavy wind; widespread damage to weak structures; large branches broken; danger to light outdoor objects.`,
  //     leadTime: "18 Hours",
  //     subcategory: "cyclone",
  //   },
  //   {
  //     term: "Typhoon - Level Winds",
  //     color: "#E63946",
  //     subtitle: "Wind Signal 4",
  //     threshold: "118-184KPH",
  //     definition: `Violent wind; uprooted trees; significant structural damage; dangerous to all outdoor activities; stay indoors.`,
  //     leadTime: "18 Hours",
  //     subcategory: "cyclone",
  //   },
  //   {
  //     term: "Super Typhoon - Level Winds",
  //     color: "#CB00CE",
  //     subtitle: "Wind Signal 5",
  //     threshold: "185KPH and above",
  //     definition: `Extremely violent winds; widespread devastation; catastrophic damage expected; emergency measures imperative.`,
  //     leadTime: "12 Hours",
  //     subcategory: "cyclone",
  //   },
  // ],
  "Precipitation": [
    {
      term: "Light Rain",
      color: "#B0E0E6",
      threshold: "2.5 mm/h",
      definition:
        "Individual drops easily identified and puddles(small muddy pools) form slowly. Small streams may flow in gutters.",
    },
    {
      term: "Moderate Rain",
      color: "#00BFFF",
      threshold: "2.5-7.5 mm/h",
      definition: "Puddles rapidly forming and down pipes flowing freely",
    },
    {
      term: "Heavy Rain",
      color: "#FACC15",
      subtitle: "Yellow Rainfall",
      threshold: "7.5-15 mm/h",
      definition:
        "The sky is overcast, there is a continuous precipitation. Falls in sheets, misty spray over hard surfaces. May cause roaring noise on roofs.",
    },
    {
      term: "Intense Rain",
      color: "#FFA500",
      subtitle: "Orange Rainfall",
      threshold: "15-30 mm/h",
      definition: "Flooding is threatening",
    },
    {
      term: "Torrential Rain",
      color: "#DC3545",
      subtitle: "Red Rainfall",
      threshold: "30 mm/h+",
      definition: "Flooding is threatening",
    },
  ],
  "UV Index": [
    {
      term: "Minimal",
      color: "#D6D6D6",
      threshold: "1-2",
      definition:
        "Wear sunglasses on bright days. In winter, reflection off snow can nearly double UV strength. If you burn easily, cover up and use sunscreen.",
    },
    {
      term: "Moderate",
      color: "#FFBC01",
      threshold: "3-5",
      definition:
        "Take precautions, such as covering and using sunscreen, if you will be outside. Stay in shade near midday when the sun is strongest.",
    },
    {
      term: "High",
      color: "#FF9000",
      threshold: "6-7",
      definition:
        "Protection against sunburn is needed. Reduce time in the sun between 11 a.m. and 4 p.m. Cover up, wear a hat and sunglasses, and use sunscreen",
    },
    {
      term: "Very High",
      color: "#F55023",
      threshold: "8-10",
      definition:
        "Take extra precautions. Unprotected skin will be damaged and can burn quickly. Try to avoid the sun between 11 a.m and 4 p.m. Otherwise, seek shade, cover up, wear a hat and sunglasses, and use sunscreen.",
    },
    {
      term: "Extreme",
      color: "#9E47CC",
      threshold: "11+",
      definition:
        "Take all precautions. unprotected skin can burn in minutes. Beachgoers should know that white sand and other bright surfaces reflect UV and will increase UV exposure. Avoid the sun between 11 a.m and 4 p.m. Seek shade, cover up, wear a hat and sunglasses, and use sunscreen.",
    },
  ],
  "Humidity": [
    {
      term: "Low Humidity",
      threshold: "0-30%",
      color: "#D97706",
      definition:
        "Air is dry; may cause dry skin, irritation to eyes and respiratory passages. Hydration recommended."
    },
    {
      term: "Moderate Humidity",
      threshold: "31-69%",
      color: "#3B82F6",
      definition:
        "Comfortable humidity range for most people; minimal effects on health or comfort."
    },
    {
      term: "High Humidity",
      threshold: "70-79%",
      color: "#F59E0B",
      definition:
        "High humidity; may feel sticky and uncomfortable. Heat stress risk increases during activity."
    },
    {
      term: "Extreme Humidity",
      threshold: "80-100%",
      color: "#DC2626",
      definition:
        "Extremely high humidity; risk of heat exhaustion or heat stroke rises significantly. Stay hydrated and limit strenuous activity."
    }
  ],

  "Pressure": [
    {
      term: "Very Low Pressure",
      threshold: "0-979 hPa",
      color: "#1D4ED8",
      definition:
        "Indicates stormy or unstable weather; low pressure often associated with rain or strong winds."
    },
    {
      term: "Low Pressure",
      threshold: "980-990 hPa",
      color: "#3B82F6",
      definition:
        "Slightly low atmospheric pressure; conditions may favor cloudiness or precipitation."
    },
    {
      term: "Normal Pressure",
      threshold: "991-1030 hPa",
      color: "#60A5FA",
      definition:
        "Average atmospheric pressure; weather generally stable and fair."
    },
    {
      term: "High Pressure",
      threshold: "1031-1040 hPa",
      color: "#93C5FD",
      definition:
        "High pressure; usually indicates clear, calm weather with minimal precipitation."
    },
    {
      term: "Very High Pressure",
      threshold: "1041 hPa+",
      color: "#BFDBFE",
      definition:
        "Extremely high pressure; very stable, dry weather. Can contribute to haze or fog under certain conditions."
    }
  ],

  "Light Intensity": [
    {
      term: "Dim Light",
      threshold: "0-50 lux",
      color: "#B0C4DE",
      definition:
        "Very low light levels; reading or detailed work may be difficult without artificial lighting."
    },
    {
      term: "Indoor Lighting",
      threshold: "51-200 lux",
      color: "#ADD8E6",
      definition:
        "Typical home or office lighting; suitable for general indoor activities."
    },
    {
      term: "Office Lighting",
      threshold: "201-500 lux",
      color: "#FFFFE0",
      definition:
        "Bright indoor environment; suitable for detailed tasks and office work."
    },
    {
      term: "Bright Indoor Lighting",
      threshold: "501-1000 lux",
      color: "#FFFF99",
      definition:
        "Very bright indoor conditions; ideal for precise visual tasks and reducing eye strain."
    },
    {
      term: "Overcast Daylight",
      threshold: "1001 - 10000 lux",
      color: "#FFFACD",
      definition:
        "Natural daylight on an overcast day; generally comfortable for outdoor activity."
    },
    {
      term: "Full Daylight",
      threshold: "10001 - 25000 lux",
      color: "#FFD700",
      definition:
        "Bright daylight without direct sun; safe for outdoor activities but consider sunglasses for comfort."
    },
    {
      term: "Direct Sunlight",
      threshold: "25001 - 100000 lux",
      color: "#FFA500",
      definition:
        "Direct sunlight; can cause glare and eye strain; UV exposure risk is high; protection recommended."
    }
  ]

};
