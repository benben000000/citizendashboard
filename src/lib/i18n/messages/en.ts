export const en = {
    common: {
      appName: "Kloudtrack",
      kloudtechLogoAlt: "KloudTrack Logo",
      units: {
        celsius: "\u00B0C",
      },
      metrics: {
        temperature: "Temperature",
        heatIndex: "Heat Index",
        humidity: "Humidity",
        pressure: "Pressure",
        wind: "Wind",
        windSpeed: "Wind Speed",
        precipitation: "This Hour's Precipitation",
        dayPrecipitation: "Today's Precipitation",
        uvIndex: "UV Index",
        light: "Light",
        lightIntensity: "Light Intensity",
      },
      chart: {
        today: "Today",
        yesterday: "Yesterday",
        value: "Value",
        highestToday: "HIGHEST TODAY",
        lowestToday: "LOWEST TODAY",
      },
      status: {
        noDataAvailable: "[NO DATA AVAILABLE]",
        retry: "RETRY",
      },
    },
    stationSelector: {
      ariaLabel: "Select station location",
      eyebrow: "Explore locations",
      detectingLocation: "Detecting location...",
      findNearestStation: "Find Nearest Station",
      pleaseWait: "Please wait...",
      useCurrentLocation: "Use your current location",
      nearest: "Nearest",
      allStations: "All Stations",
    },
    language: {
      label: "Language",
    },
    routeTransition: {
      weather: "Loading weather",
      waterLevel: "Loading water level",
      prediction: "Loading prediction",
      fallback: "Loading",
      srOnly: "Please wait while the next page loads.",
    },
    dashboard: {
      current: {
        todayAt: "today at",
      },
      actions: {
        stepsToStaySafe: "Steps to stay safe",
      },
      info: {
        contact: "Contact",
        email: "Email",
        exploreYourArea: "Explore your area",
      },
      metrics: {
        title: "Weather Metrics",
        guides: "Weather Term Guides",
        guidesAriaLabel: "Weather Term Guides",
        thisHour: "This hour",
      },
      chart: {
        title: "24 Hour Overview",
        unavailableForMetric: "[NO HISTORICAL CHART AVAILABLE FOR THIS METRIC]",
      },
      map: {
        tokenMissing: "Map unavailable: Mapbox token not configured",
      },
      ctaBanner: {
        title: "Build Resilience With Us",
        description:
          "Learn how our real-time, hyper-local weather platform supports community preparedness and how we can deploy a tailored system for your area.",
        learnMore: "Learn More",
        contactUs: "Contact Us",
      },
    },
    waterLevel: {
      metrics: {
        title: "Metrics",
        waterLevel: "Water Level",
        changeFromToday: "Change from today",
        trend: "Trend",
      },
      chart: {
        title: "24 Hour Overview",
        notEnoughHistory: "Not enough history is available yet to draw the trend.",
        bridgeReference: "Bridge",
      },
      map: {
        tokenMissing: "Map unavailable: Mapbox token not configured",
        stationFallback: "Water level station",
      },
      stationSelector: {
        unavailable: "No water-level station is available.",
      },
      warnings: {
        allUnavailable: "Water-level readings are temporarily unavailable.",
        currentUnavailable:
          "The current reading is unavailable, but the historical trend is still shown.",
        historyUnavailable:
          "The historical trend is temporarily unavailable, but the current reading is still shown.",
      },
      trend: {
        rising: "Rising",
        falling: "Falling",
        stable: "Stable",
        unknown: "No trend",
      },
    },
    prediction: {
      nowcastingPill: "Sub-Second Stream Nowcasting (17.14 μs ODE)",
      temperature: "Temperature",
      todayAt: "today at",
      refresh: "Refresh",
      horizonLabel: "Horizon:",
      cards: {
        heatIndex: "Heat Index",
        windPressure: "Wind & Pressure",
        rainChance: "Chance of Rain",
        waterLevelFlood: "Water Level & Flood Risk",
        precipitation: "Precipitation",
        floodRisk: "Flood Risk in Area",
        humidity: "Humidity",
        uvIndex: "UV Index",
        yes: "YES",
        no: "NO",
        possible: "POSSIBLE",
        lowRisk: "Low Risk",
        highRisk: "High Risk",
        moderateRisk: "Moderate Risk",
        highHumidity: "High Humidity",
        dryAir: "Dry Air",
        normalHumidity: "Optimal Humidity",
        uvVeryHigh: "Very High / Seek Shade",
        uvHigh: "High / Sun Protection",
        uvModerate: "Moderate Exposure",
        uvLow: "Low / Minimal",
      },
      heatIndexCategories: {
        extremeDanger: "Extreme Danger",
        danger: "Danger",
        extremeCaution: "Extreme Caution",
        caution: "Caution",
        comfortable: "Comfortable",
      },
      riskBadges: {
        normal: "Normal",
        advisory: "Alert",
        warning: "Warning",
        critical: "Critical",
      },
      waterStatuses: {
        normal: "Safe Level",
        advisory: "Monitor / Alert",
        warning: "Elevated / Warning",
        critical: "Critical Flooding",
      },
      rainLikelihood: {
        heavy: "Heavy Rain Expected",
        scattered: "Scattered Rain / Showers",
        low: "Low Rain Chance",
      },
      advisories: {
        normal: "Warm weather conditions. Keep hydrated and take shade if active outdoors.",
        rainNormal: "Passing light drizzle. Bring light rain gear if heading outdoors.",
        advisory: "Scattered rain showers. Monitor low-lying areas and road drainage as rain continues.",
        warning: "Heavy rainfall detected. Prepare essentials and stay alert for localized street flooding.",
        critical: "Critical heavy storm and flooding risk. Follow immediate safety notices from local authorities.",
      },
      peakSection: {
        crestTitle: "Road & Flood Passability",
        crestSubtitle: "Is it safe to travel or will roads flood in your area?",
        umbrellaTitle: "Rain & Umbrella Guide",
        umbrellaSubtitle: "Should you bring an umbrella or raincoat today?",
        mountainTitle: "Mountain Flash Flood Alert",
        mountainSubtitle: "Water runoff risk from nearby mountains and watersheds",
        
        passableSafe: "SAFE TO PASS / ROADS CLEAR",
        passableCaution: "CAUTION: ROADS MAY BE WET",
        passableDanger: "DANGER: FLOODED / DO NOT PASS",
        
        passableSafeDesc: "Roads are dry and river levels are safe. Clear for all motorbikes, cars, and pedestrians.",
        passableCautionDesc: "Ankle to gutter-deep water in low-lying roads. Small vehicles and motorbikes take extra care.",
        passableDangerDesc: "Knee to waist-deep flood! High risk for light vehicles and motorbikes. Take alternate routes.",

        umbrellaNoRain: "CLEAR / NO UMBRELLA NEEDED",
        umbrellaLight: "BRING AN UMBRELLA (PASSING SHOWERS)",
        umbrellaHeavy: "HEAVY DOWNPOUR COMING! BRING RAINGEAR",
        
        umbrellaNoRainDesc: "Dry and clear skies. No rain protection needed.",
        umbrellaLightDesc: "Light passing drizzle expected around {time} lasting ~{duration}. Carry an umbrella.",
        umbrellaHeavyDesc: "Sudden heavy cloudburst around {time}. Bring a raincoat/umbrella and seek shelter if driving.",

        mountainSafe: "MOUNTAIN RUNOFF SAFE",
        mountainCaution: "ACTIVE MOUNTAIN RAIN (MONITOR)",
        mountainDanger: "FLASH FLOOD SURGE FROM MOUNTAINS!",
        
        mountainSafeDesc: "No heavy rain in the mountains. Safe for communities near mountain slopes and riverbanks.",
        mountainCautionDesc: "Rain detected in the mountains. Watch out for delayed rising river water in low areas.",
        mountainDangerDesc: "Heavy storm in the mountains! Flash floods may surge downstream even if it's not raining in your area.",

        expectedAt: "Projected peak around {time}",
        clearanceNormal: "{clearance}m clearance below Critical Flood Stage ({critical}m)",
        clearanceExceeded: "Exceeds Critical Flood Level by {clearance}m",
        watershedTitle: "Upstream Watershed & Inflow",
        watershedSubtitle: "Mountain runoff & upstream station accumulation",
        inflowStatus: "Upstream Inflow Status",
        inflowNormal: "Normal / Gentle Inflow",
        inflowElevated: "Elevated Runoff",
        inflowCritical: "Surge Runoff Detected",
        mountainRain: "Upstream Rain Rate",
        runoffDescriptionNormal: "No heavy rainfall detected upstream that would cause sudden river surges.",
        runoffDescriptionElevated: "Active rainfall detected in upstream mountains. Monitor river level for delayed runoff surge.",
        thresholdLabels: {
          normal: "Safe (<{val}m)",
          advisory: "Gutter / Ankle",
          warning: "Knee Deep",
          critical: "Critical ({val}m)",
        },
        burstTitle: "Rain & Umbrella Guide",
        burstSubtitle: "Should you bring an umbrella or raincoat today?",
        leadHorizonTitle: "Forecast Lead Horizon",
        leadHorizonSubtitle: "Dynamic continuous-time ODE projection timeframe",
        burstTypes: {
          sudden_heavy: "Heavy Downpour Alert",
          short_burst_heavy: "Short Burst of Heavy Downpour",
          sudden_light: "Light Passing Showers",
          short_burst_light: "Short Passing Drizzle",
          none: "No Rain Expected",
        },
        burstBadges: {
          sudden_heavy: "Heavy Rain",
          short_burst_heavy: "Heavy Burst",
          sudden_light: "Passing Showers",
          short_burst_light: "Light Drizzle",
          none: "Clear Skies",
        },
        burstLabels: {
          rate: "Rain Rate",
          duration: "Expected Duration",
          onset: "Expected Time",
          radar: "Doppler Radar",
          cloud: "Satellite Cloud",
          minutes: "{min} mins",
          stable: "Clear skies. No sudden downpours expected.",
        },
      },
    },
    terminology: {
      backToDashboard: "Back to dashboard",
      title: "Weather Term Guides",
      subtitle: "Understanding weather metrics and their categories.",
      understanding: "Understanding {metric}",
      noReferences: "No references available.",
    },
    weatherWarnings: {
      common: {
        normal: "Normal",
      },
      warnings: {
        heatIndexCaution: {
          term: "Stay Hydrated",
          warningLevel: "Caution",
          suggestedAction:
            "Mild discomfort possible. Drink water regularly and take light breaks in the shade.",
        },
        heatIndexExtremeCaution: {
          term: "Rest Often",
          warningLevel: "Extreme Caution",
          suggestedAction:
            "Heat cramps and exhaustion possible. Limit prolonged outdoor activity and stay cool.",
        },
        heatIndexDanger: {
          term: "Avoid Outdoors",
          warningLevel: "Danger",
          suggestedAction:
            "Heat exhaustion likely and heat stroke possible. Avoid outdoor exposure and stay in a cool place.",
        },
        heatIndexExtremeDanger: {
          term: "Life-Threatening",
          warningLevel: "Extreme Danger",
          suggestedAction:
            "Heat stroke is highly likely. Stay indoors, avoid all exertion, and seek immediate help if symptoms appear.",
        },
        windTropicalDepression: {
          term: "Strong Winds",
          warningLevel: "Tropical Depression - Level Winds",
        },
        windTropicalStorm: {
          term: "Damaging Winds",
          warningLevel: "Tropical Storm - Level Winds",
        },
        windSevereTropicalStorm: {
          term: "Destructive Winds",
          warningLevel: "Severe Tropical Storm - Level Winds",
        },
        windTyphoon: {
          term: "Typhoon",
          warningLevel: "Typhoon - Level Winds",
        },
        windSuperTyphoon: {
          term: "Super Typhoon",
          warningLevel: "Super Typhoon - Level Winds",
        },
        precipitationLightRain: {
          term: "Light Rain",
          warningLevel: "Light Rain",
          suggestedAction:
            "Light rain is present. Bring rain protection and allow a little extra travel time.",
        },
        precipitationModerateRain: {
          term: "Moderate Rain",
          warningLevel: "Moderate Rain",
          suggestedAction:
            "Moderate rain may make roads slippery. Slow down, protect electronics, and monitor updates.",
        },
        precipitationHeavyRain: {
          term: "Heavy Rain",
          warningLevel: "Heavy Rains",
          suggestedAction:
            "Heavy rain can reduce visibility and cause pooling water. Delay non-essential travel when possible.",
        },
        precipitationFloodRisk: {
          term: "Flood Risk",
          warningLevel: "Intense Rain",
          suggestedAction:
            "Intense rain may trigger flooding. Move valuables higher and avoid flood-prone routes.",
        },
        precipitationSevereFlooding: {
          term: "Severe Flooding",
          warningLevel: "Torrential Rain",
          suggestedAction:
            "Torrential rain can be dangerous. Stay indoors, avoid floodwater, and prepare to evacuate if advised.",
        },
        uvWearSunscreen: {
          term: "Wear Sunscreen",
          warningLevel: "Moderate",
        },
        uvSeekShade: {
          term: "Seek Shade",
          warningLevel: "High",
        },
        uvAvoidSun: {
          term: "Avoid the Sun",
          warningLevel: "Very High",
        },
        uvStayInside: {
          term: "Stay Inside",
          warningLevel: "Extreme",
        },
        temperatureCool: {
          term: "Cool",
          warningLevel: "Cool",
          suggestedAction:
            "Cool conditions are present. Wear comfortable layers if staying outside for long periods.",
        },
        temperatureWarm: {
          term: "Warm",
          warningLevel: "Moderate",
          suggestedAction:
            "Warm conditions are present. Drink water regularly and take breaks when active outdoors.",
        },
        temperatureGettingHot: {
          term: "Getting Hot",
          warningLevel: "Warm",
          suggestedAction:
            "Temperature is getting hot. Stay hydrated and avoid prolonged sun exposure.",
        },
        temperatureVeryHot: {
          term: "Very Hot",
          warningLevel: "Hot",
          suggestedAction:
            "Very hot conditions can cause heat stress. Seek shade and limit strenuous activity.",
        },
        temperatureDangerous: {
          term: "Dangerous",
          warningLevel: "Extreme Heat",
          suggestedAction:
            "Dangerous heat is present. Stay in a cool place and watch for heat illness symptoms.",
        },
        humidityVeryDry: {
          term: "Very Dry",
          warningLevel: "Low Humidity",
        },
        humidityComfortable: {
          term: "Comfortable",
          warningLevel: "Moderate Humidity",
        },
        humidityHumid: {
          term: "Humid",
          warningLevel: "High Humidity",
        },
        humidityVeryHumid: {
          term: "Very Humid",
          warningLevel: "Extreme Humidity",
        },
        pressureStormLikely: {
          term: "Storm Likely",
          warningLevel: "Very Low Pressure",
        },
        pressureRainPossible: {
          term: "Rain Possible",
          warningLevel: "Low Pressure",
        },
        pressureFairWeather: {
          term: "Fair Weather",
          warningLevel: "Normal Pressure",
        },
        pressureClearCalm: {
          term: "Clear & Calm",
          warningLevel: "High Pressure",
        },
        pressureVeryDryClear: {
          term: "Very Dry & Clear",
          warningLevel: "Very High Pressure",
        },
        lightVeryDim: {
          term: "Very Dim",
          warningLevel: "Dim Light",
        },
        lightIndoor: {
          term: "Indoor Light",
          warningLevel: "Indoor Lighting",
        },
        lightBrightIndoors: {
          term: "Bright Indoors",
          warningLevel: "Office Lighting",
        },
        lightVeryBright: {
          term: "Very Bright",
          warningLevel: "Bright Indoor Lighting",
        },
        lightCloudyDay: {
          term: "Cloudy Day",
          warningLevel: "Overcast Daylight",
        },
        lightBrightDay: {
          term: "Bright Day",
          warningLevel: "Full Daylight",
        },
        lightDirectSun: {
          term: "Direct Sun",
          warningLevel: "Direct Sunlight",
        },
      },
      actions: {
        // Heat Index Actions

        // Heat Index Caution
        drinkWaterOften: "Drink water more often, even if not thirsty.",
        takeShadeBreaks:
          "Take short breaks in the shade or indoors if outside for long periods.",
        wearLightClothing: "Wear light-colored, loose-fitting clothing.",
        avoidHottestHours:
          "Avoid too much sun exposure during the hottest hours of the day.",
        watchHeatStress:
          "Watch for early signs of heat stress such as tiredness, heavy sweating, or dizziness.",
        
        // Heat Index Extreme Caution
        limitOutdoorActivity: "Limit prolonged outdoor activity.",
        stayShadedVentilated:
          "Stay in shaded or well-ventilated areas as much as possible.",
        drinkWaterFrequently: "Drink water frequently.",
        useSunProtection:
          "Use umbrellas, hats, or other sun protection when outside.",
        stopAndCoolDown:
          "Stop activity and cool down immediately if feeling weak, nauseated, or dizzy.",
        
        // Heat Index Danger
        avoidOutdoorExposure: "Avoid unnecessary outdoor exposure.",
        stayCoolestIndoors: "Stay indoors in the coolest available place.",
        drinkWaterThroughoutDay: "Drink water regularly throughout the day.",
        monitorHeatIllness:
          "Closely monitor for symptoms of heat exhaustion or heat stroke.",
        seekMedicalHelp:
          "Seek medical help promptly if someone becomes dizzy, faints, vomits, or stops sweating.",
        
        // Heat Index Extreme Danger
        stayIndoorsPossible: "Stay indoors as much as possible.",
        avoidOutdoorActivity:
          "Avoid outdoor activity except when absolutely necessary.",
        stayHydratedCool:
          "Keep hydrated and remain in a cool or air-conditioned place if available.",
        watchHeatStroke:
          "Be alert for heat stroke symptoms: dizziness, loss of consciousness, very high body temperature, or seizures.",
        callEmergencyHelp:
          "Call emergency services or seek urgent medical help if heat stroke is suspected.",

        // Precipitation Actions
        // Light Rain
        bringRainProtection: "Bring an umbrella or raincoat before going out.",
        checkWeatherUpdates: "Check weather updates before traveling.",
        wearNonSlipFootwear: "Wear footwear with good grip on wet roads.",
        coverImportantItems: "Cover bags, gadgets, and documents from rain.",
        allowExtraTravelTime: "Leave a little earlier for possible traffic delays.",

        // Moderate Rain
        avoidSlipperyAreas: "Avoid slippery sidewalks and flooded corners.",
        secureOutdoorItems: "Move clothes, tools, and outdoor items under cover.",
        monitorRainConditions: "Monitor local advisories while rain continues.",
        protectElectronics: "Keep phones and electronics in waterproof bags.",
        slowDownOnRoads: "Drive slower and keep distance from other vehicles.",

        // Heavy Rain
        delayNonEssentialTravel: "Delay unnecessary trips until rain becomes lighter.",
        avoidLowLyingRoads: "Avoid low-lying roads and known flood areas.",
        prepareEmergencyLighting: "Prepare flashlights and charge mobile devices early.",
        watchForPoorVisibility: "Use caution on roads with poor visibility.",
        monitorFamilyMembers: "Check on children and elderly family members regularly.",

        // Flood Risk
        moveValuablesHigher: "Move important items and appliances to higher places.",
        prepareEmergencyBag: "Prepare emergency supplies and extra drinking water.",
        avoidFloodProneRoutes: "Avoid riversides, underpasses, and flood-prone roads.",
        followLocalAdvisories: "Follow updates from local authorities and responders.",
        checkEvacuationOptions: "Identify safe routes and nearby evacuation areas.",

        // Severe Flooding
        stayIndoorsIfPossible: "Stay indoors unless travel is absolutely necessary.",
        avoidFloodwaters: "Avoid walking or driving through floodwater.",
        assistVulnerablePeople: "Assist children, seniors, and persons needing support.",
        prepareForEvacuation: "Prepare to evacuate immediately if advised by authorities.",
        contactEmergencyServices: "Call local emergency services if immediate help is needed.",
        
        // Default Actions
        defaultDrinkWater:
          "Drink water regularly, especially if outdoors for long periods.",
        defaultWearComfortableClothing: "Wear light, comfortable clothing.",
        defaultCheckUpdates: "Continue checking if temperature is rising.",
        defaultStayCool: "Stay in shaded or cool areas when possible.",
      },
    },
} as const;
