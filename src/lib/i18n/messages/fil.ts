import type { Messages } from "@/lib/i18n/translations";

export const fil = {
  common: {
    appName: "Kloudtrack",
    kloudtechLogoAlt: "KloudTrack Logo",
    units: {
      celsius: "\u00B0C",
    },
    metrics: {
      temperature: "Temperatura",
      heatIndex: "Damang Init",
      humidity: "Halumigmig",
      pressure: "Pressure",
      wind: "Hangin",
      windSpeed: "Bilis ng Hangin",
      precipitation: "Ulan Ngayong Oras",
      dayPrecipitation: "Ulan Ngayong Araw",
      uvIndex: "UV Index",
      light: "Liwanag",
      lightIntensity: "Lakas ng Liwanag",
    },
    chart: {
      today: "Ngayon",
      yesterday: "Kahapon",
      value: "Halaga",
      highestToday: "PINAKAMATAAS",
      lowestToday: "PINAKAMABABA",
    },
    status: {
      noDataAvailable: "[WALANG AVAILABLE NA DATOS]",
      retry: "SUBUKAN ULIT",
    },
  },
  stationSelector: {
    ariaLabel: "Pumili ng lokasyon ng station",
    eyebrow: "Tingnan ang mga lokasyon",
    detectingLocation: "Hinahanap ang lokasyon...",
    findNearestStation: "Hanapin ang Pinakamalapit na Station",
    pleaseWait: "Sandali lang...",
    useCurrentLocation: "Gamitin ang kasalukuyang lokasyon",
    nearest: "Pinakamalapit",
    allStations: "Lahat ng Stations",
  },
  language: {
    label: "Wika",
  },
  routeTransition: {
    weather: "Naglo-load ng panahon",
    waterLevel: "Naglo-load ng antas ng tubig",
    prediction: "Naglo-load ng prediksyon",
    fallback: "Naglo-load",
    srOnly: "Pakihintay habang naglo-load ang susunod na pahina.",
  },
  dashboard: {
    current: {
      todayAt: "ngayong",
    },
    actions: {
      stepsToStaySafe: "Mga hakbang para manatiling ligtas",
    },
    info: {
      contact: "Contact",
      email: "Email",
      exploreYourArea: "Siyasatin ang mga lugar",
    },
    metrics: {
      title: "Weather Metrics",
      guides: "Weather Term Guides",
      guidesAriaLabel: "Weather Term Guides",
      thisHour: "Ngayong oras",
    },
    chart: {
      title: "24 Oras na Sumaryo",
      unavailableForMetric: "[WALANG HISTORICAL CHART PARA SA METRIC NA ITO]",
    },
    map: {
      tokenMissing: "Hindi available ang mapa: walang naka-configure na Mapbox token",
    },
    ctaBanner: {
      title: "Handa sa Hamon ng Panahon",
      description:
        "Alamin kung paano nakakatulong ang aming real-time at hyper-local na weather platform sa paghahanda ng komunidad, at kung paano kami makakapag-deploy ng angkop na solusyon para sa inyong lugar.",
      learnMore: "Matuto Pa",
      contactUs: "Makipag-ugnayan",
    },
  },
  waterLevel: {
    metrics: {
      title: "Weather Metrics",
      waterLevel: "Antas ng Tubig",
      changeFromToday: "Pagbabago ngayong araw",
      trend: "Takbo",
    },
    chart: {
      title: "24 Oras na Sumaryo",
      notEnoughHistory: "Hindi pa sapat ang history para maipakita ang trend.",
      bridgeReference: "Tulay",
    },
    map: {
      tokenMissing: "Hindi available ang mapa: walang naka-configure na Mapbox token",
      stationFallback: "Station ng antas ng tubig",
    },
    stationSelector: {
      unavailable: "Walang available na station ng antas ng tubig.",
    },
    warnings: {
      allUnavailable: "Pansamantalang hindi available ang mga reading ng antas ng tubig.",
      currentUnavailable:
        "Hindi available ang kasalukuyang reading, pero ipinapakita pa rin ang dating trend.",
      historyUnavailable:
        "Pansamantalang hindi available ang dating trend, pero ipinapakita pa rin ang kasalukuyang reading.",
    },
    trend: {
      rising: "Tumataas",
      falling: "Bumababa",
      stable: "Hindi nagbabago",
      unknown: "Walang trend",
    },
  },
  prediction: {
    nowcastingPill: "Sub-Second Stream Nowcasting (17.14 μs ODE)",
    temperature: "Temperatura",
    todayAt: "ngayong",
    refresh: "I-refresh",
    horizonLabel: "Abot-Tanaw:",
    cards: {
      heatIndex: "Damang Init",
      windPressure: "Hangin at Presyon",
      rainChance: "Tsansa ng Ulan",
      waterLevelFlood: "Antas ng Tubig at Baha",
    },
    heatIndexCategories: {
      extremeDanger: "Matinding Panganib",
      danger: "Panganib",
      extremeCaution: "Sobrang Alinsangan",
      caution: "Alinsangan",
      comfortable: "Komportable",
    },
    riskBadges: {
      normal: "Normal",
      advisory: "Alerto",
      warning: "Babala",
      critical: "Kritikal",
    },
    waterStatuses: {
      normal: "Ligtas na Antas",
      advisory: "Bantayan / Alert",
      warning: "Mataas / Warning",
      critical: "Kritikal na Baha",
    },
    rainLikelihood: {
      heavy: "May Malakas na Ulan",
      scattered: "May Pag-ambon / Ulan",
      low: "Mababang Tsansa",
    },
    advisories: {
      normal: "Mainit-init ang panahon. Regular na uminom ng tubig at magpahinga kung aktibo sa labas.",
      rainNormal: "Mahinang ambon. Magdala ng payong kung lalabas ng bahay.",
      advisory: "May kalat-kalat na pag-ulan. Bantayan ang mga mabababang lugar at daluyan ng tubig.",
      warning: "Malakas na buhos ng ulan. Ihanda ang mga gamit at maging alerto sa posibleng pagbaha.",
      critical: "Nasa kritikal na antas ang ulan at baha. Sundin ang agarang abiso sa kaligtasan mula sa lokal na awtoridad.",
    },
    peakSection: {
      crestTitle: "Inaasahang Pinakamataas na Antas ng Tubig",
      crestSubtitle: "Tantiyang pinakamataas na lebel ng tubig sa buong lead horizon",
      expectedAt: "Inaasahang peak bandang {time}",
      clearanceNormal: "Ligtas pa ng {clearance}m bago umabot sa Kritikal na Lebel ({critical}m)",
      clearanceExceeded: "Lumampas ng {clearance}m sa Kritikal na Lebel ng Baha",
      watershedTitle: "Daloy sa Watershed at Kabundukan",
      watershedSubtitle: "Pag-ulan sa kabundukan at papasok na tubig sa ilog",
      inflowStatus: "Kasalukuyang Daloy ng Tubig",
      inflowNormal: "Normal na Daloy",
      inflowElevated: "Mabilis na Daloy",
      inflowCritical: "Malakas na Bugso",
      mountainRain: "Ulan sa Itaas ng Bundok",
      runoffDescriptionNormal: "Walang namumuong malakas na pagbuhos ng ulan sa kabundukan na magdudulot ng biglaang pagtaas ng tubig.",
      runoffDescriptionElevated: "May pag-ulan sa kabundukan. Bantayan ang ilog sa inaasahang pagbaba ng tubig mula sa itaas.",
      thresholdLabels: {
        normal: "Normal",
        advisory: "Alerto",
        warning: "Babala",
        critical: "Kritikal",
      },
      burstTitle: "Prediksyon sa Biglaang Pagbuhos ng Ulan",
      burstSubtitle: "Pagsusuri gamit ang Doppler Radar at Himawari Satellite",
      leadHorizonTitle: "Oras ng Prediksyon (Lead Horizon)",
      leadHorizonSubtitle: "Dynamic na pagtatantiya sa susunod na mga oras",
      burstTypes: {
        sudden_heavy: "Natukoy na Biglaang Malakas na Ulan",
        short_burst_heavy: "Maikling Bugso ng Malakas na Buhos",
        sudden_light: "Biglaang Mahinang Pag-ambon / Ulan",
        short_burst_light: "Maikling Bugso ng Mahinang Ulan",
        none: "Walang Biglaang Ulan na Inaasahan",
      },
      burstBadges: {
        sudden_heavy: "Malakas",
        short_burst_heavy: "Malakas",
        sudden_light: "Mahina",
        short_burst_light: "Mahina",
        none: "Normal",
      },
      burstLabels: {
        rate: "Tindi ng Pagbuhos",
        duration: "Tantiyang Tagal",
        onset: "Oras ng Pagbuhos",
        radar: "Doppler Radar",
        cloud: "Convective Index",
        minutes: "{min} minuto",
        stable: "Kalmado ang atmospera. Walang inaasahang biglaang buhos ng ulan.",
      },
    },
  },
  terminology: {
    backToDashboard: "Bumalik sa dashboard",
    title: "Gabay sa mga Terminong Pang-Panahon",
    subtitle: "Pag-unawa sa weather metrics at mga category nito.",
    understanding: "Pag-unawa sa {metric}",
    noReferences: "Walang available na references.",
  },
  weatherWarnings: {
    common: {
      normal: "Normal",
    },
    warnings: {
      heatIndexCaution: {
        term: "Uminom ng Tubig",
        warningLevel: "Pag-iingat",
        suggestedAction:
          "Maaaring makaramdam ng kaunting discomfort. Regular na uminom ng tubig at magpahinga sa lilim.",
      },
      heatIndexExtremeCaution: {
        term: "Magpahinga Madalas",
        warningLevel: "Matinding Pag-iingat",
        suggestedAction:
          "Posible ang heat cramps at heat exhaustion. Bawasan ang matagal na outdoor activity at manatiling presko.",
      },
      heatIndexDanger: {
        term: "Iwas sa Labas",
        warningLevel: "Panganib",
        suggestedAction:
          "Malamang ang heat exhaustion at posible ang heat stroke. Iwasan ang outdoor exposure at manatili sa malamig na lugar.",
      },
      heatIndexExtremeDanger: {
        term: "Delikado sa Buhay",
        warningLevel: "Labis na Panganib",
        suggestedAction:
          "Mataas ang posibilidad ng heat stroke. Manatili sa loob, iwasan ang pagod, at humingi agad ng tulong kung may sintomas.",
      },
      windTropicalDepression: {
        term: "Malakas na Hangin",
        warningLevel: "Tropical Depression - Level Winds",
      },
      windTropicalStorm: {
        term: "Nakakasirang Hangin",
        warningLevel: "Tropical Storm - Level Winds",
      },
      windSevereTropicalStorm: {
        term: "Mapaminsalang Hangin",
        warningLevel: "Severe Tropical Storm - Level Winds",
      },
      windTyphoon: {
        term: "Bagyo",
        warningLevel: "Typhoon - Level Winds",
      },
      windSuperTyphoon: {
        term: "Super Typhoon",
        warningLevel: "Super Typhoon - Level Winds",
      },
      precipitationLightRain: {
        term: "Mahinang Ulan",
        warningLevel: "Light Rain",
        suggestedAction:
          "May mahinang ulan. Magdala ng payong o kapote at maglaan ng dagdag na oras sa biyahe.",
      },
      precipitationModerateRain: {
        term: "Katamtamang Ulan",
        warningLevel: "Moderate Rain",
        suggestedAction:
          "Maaaring maging madulas ang daan. Magdahan-dahan, protektahan ang gadgets, at subaybayan ang updates.",
      },
      precipitationHeavyRain: {
        term: "Malakas na Ulan",
        warningLevel: "Heavy Rains",
        suggestedAction:
          "Maaaring humina ang visibility at magkaroon ng naipong tubig. Ipagpaliban ang hindi mahalagang biyahe kung maaari.",
      },
      precipitationFloodRisk: {
        term: "Napakalakas na Ulan",
        warningLevel: "Intense Rain",
        suggestedAction:
          "Maaaring magdulot ng baha ang malakas na ulan. Itaas ang mahahalagang gamit at iwasan ang bahain na ruta.",
      },
      precipitationSevereFlooding: {
        term: "Matinding Pagulan",
        warningLevel: "Torrential Rain",
        suggestedAction:
          "Delikado ang matinding ulan. Manatili sa loob, iwasan ang baha, at maghanda kung may abiso ng evacuation.",
      },
      uvWearSunscreen: {
        term: "Mag-sunscreen",
        warningLevel: "Moderate",
      },
      uvSeekShade: {
        term: "Humanap ng Lilim",
        warningLevel: "High",
      },
      uvAvoidSun: {
        term: "Iwasan ang Araw",
        warningLevel: "Very High",
      },
      uvStayInside: {
        term: "Manatili sa Loob",
        warningLevel: "Extreme",
      },
      temperatureCool: {
        term: "Malamig",
        warningLevel: "Cool",
        suggestedAction:
          "Malamig ang kondisyon. Magsuot ng komportableng patong na damit kung matagal sa labas.",
      },
      temperatureWarm: {
        term: "Mainit-init",
        warningLevel: "Moderate",
        suggestedAction:
          "Mainit-init ang panahon. Regular na uminom ng tubig at magpahinga kung aktibo sa labas.",
      },
      temperatureGettingHot: {
        term: "Umiinit",
        warningLevel: "Warm",
        suggestedAction:
          "Umiinit ang temperatura. Manatiling hydrated at iwasan ang matagal na pagbibilad sa araw.",
      },
      temperatureVeryHot: {
        term: "Napakainit",
        warningLevel: "Hot",
        suggestedAction:
          "Maaaring magdulot ng heat stress ang sobrang init. Humanap ng lilim at bawasan ang mabibigat na gawain.",
      },
      temperatureDangerous: {
        term: "Delikado",
        warningLevel: "Extreme Heat",
        suggestedAction:
          "Delikado ang init. Manatili sa malamig na lugar at bantayan ang sintomas ng heat illness.",
      },
      humidityVeryDry: {
        term: "Tuyong-tuyo",
        warningLevel: "Low Humidity",
      },
      humidityComfortable: {
        term: "Komportable",
        warningLevel: "Moderate Humidity",
      },
      humidityHumid: {
        term: "Maalinsangan",
        warningLevel: "High Humidity",
      },
      humidityVeryHumid: {
        term: "Sobrang Alinsangan",
        warningLevel: "Extreme Humidity",
      },
      pressureStormLikely: {
        term: "Posibleng Bagyo",
        warningLevel: "Very Low Pressure",
      },
      pressureRainPossible: {
        term: "Posibleng Ulan",
        warningLevel: "Low Pressure",
      },
      pressureFairWeather: {
        term: "Maayos ang Panahon",
        warningLevel: "Normal Pressure",
      },
      pressureClearCalm: {
        term: "Maaliwalas at Kalma",
        warningLevel: "High Pressure",
      },
      pressureVeryDryClear: {
        term: "Tuyo at Maaliwalas",
        warningLevel: "Very High Pressure",
      },
      lightVeryDim: {
        term: "Madilim",
        warningLevel: "Dim Light",
      },
      lightIndoor: {
        term: "Ilaw sa Loob",
        warningLevel: "Indoor Lighting",
      },
      lightBrightIndoors: {
        term: "Maliwanag sa Loob",
        warningLevel: "Office Lighting",
      },
      lightVeryBright: {
        term: "Napakaliwanag",
        warningLevel: "Bright Indoor Lighting",
      },
      lightCloudyDay: {
        term: "Makulimlim",
        warningLevel: "Overcast Daylight",
      },
      lightBrightDay: {
        term: "Maliwanag na Araw",
        warningLevel: "Full Daylight",
      },
      lightDirectSun: {
        term: "Direktang Araw",
        warningLevel: "Direct Sunlight",
      },
    },
    actions: {
      // Heat Index Actions

      // Heat Index Caution
      drinkWaterOften: "Uminom ng tubig nang mas madalas, kahit hindi nauuhaw.",
      takeShadeBreaks:
        "Magpahinga sandali sa lilim o sa loob kung matagal nasa labas.",
      wearLightClothing: "Magsuot ng magaan at maluwag na damit, mas mabuti kung light-colored.",
      avoidHottestHours:
        "Iwasan ang matagal na pagbibilad sa araw lalo na sa pinakamainit na oras ng araw.",
      watchHeatStress:
        "Bantayan ang mga maagang senyales ng heat stress tulad ng pagkapagod, labis na pagpapawis, o pagkahilo.",
      
      // Heat Index Extreme Caution
      limitOutdoorActivity: "Limitahan ang matagal na pananatili o gawain sa labas.",
      stayShadedVentilated:
        "Manatili sa lilim o sa lugar na may maayos na ventilation hangga't maaari.",
      drinkWaterFrequently: "Madalas uminom ng tubig.",
      useSunProtection:
        "Gumamit ng payong, sombrero, o iba pang proteksyon sa araw kapag nasa labas.",
      stopAndCoolDown:
        "Itigil agad ang ginagawa at magpalamig kung nakararamdam ng panghihina, pagsusuka, o pagkahilo.",

      // Heat Index Danger
      avoidOutdoorExposure: "Iwasan ang hindi kailangang paglabas.",
      stayCoolestIndoors: "Manatili sa pinakamalamig na bahagi ng loob ng bahay o gusali.",
      drinkWaterThroughoutDay: "Regular na uminom ng tubig sa buong araw.",
      monitorHeatIllness:
        "Bantayang mabuti ang sintomas ng heat exhaustion o heat stroke.",
      seekMedicalHelp:
        "Humingi agad ng medical help kung may nahihilo, nahihimatay, nagsusuka, o tumitigil ang pagpapawis.",

      // Heat Index Extreme Danger
      stayIndoorsPossible: "Manatili sa loob hangga't maaari.",
      avoidOutdoorActivity:
        "Iwasan ang paglabas maliban kung talagang kinakailangan.",
      stayHydratedCool:
        "Uminom ng maraming tubig at manatili sa malamig o naka-aircon na lugar kung mayroon.",
      watchHeatStroke:
        "Maging alerto sa sintomas ng heat stroke: pagkahilo, pagkawala ng malay, sobrang taas na body temperature, o seizure.",
      callEmergencyHelp:
        "Tumawag sa emergency services o humingi ng medikal na saklolo kung pinaghihinalaang may heat stroke.",

      // Precipitation Actions
      // Light Rain
      bringRainProtection: "Magdala ng payong o kapote bago lumabas.",
      checkWeatherUpdates: "Tingnan ang weather updates bago bumiyahe.",
      wearNonSlipFootwear: "Magsuot ng sapatos na hindi madulas sa basang kalsada.",
      coverImportantItems: "Takpan ang bag, gadgets, at importanteng gamit sa ulan.",
      allowExtraTravelTime: "Umalis nang mas maaga para sa posibleng traffic.",

      // Moderate Rain
      avoidSlipperyAreas: "Iwasan ang madulas na daan at may bahang gilid.",
      secureOutdoorItems: "Ilipat sa masisilungan ang sinampay at outdoor gamit.",
      monitorRainConditions: "Subaybayan ang mga abiso habang tuloy ang ulan.",
      protectElectronics: "Ilagay sa waterproof na lalagyan ang gadgets at phone.",
      slowDownOnRoads: "Magdahan-dahan sa pagmamaneho sa basang kalsada.",

      // Heavy Rain
      delayNonEssentialTravel: "Ipagpaliban muna ang hindi importanteng biyahe.",
      avoidLowLyingRoads: "Iwasan ang mababang lugar at mga kalsadang bahain.",
      prepareEmergencyLighting: "Ihanda ang flashlight at i-charge ang mga device.",
      watchForPoorVisibility: "Mag-ingat sa kalsadang mahina ang visibility.",
      monitorFamilyMembers: "Kumustahin palagi ang bata at matatandang kasama sa bahay.",

      // Flood Risk
      moveValuablesHigher: "Itaas ang mga importanteng gamit at appliances.",
      prepareEmergencyBag: "Maghanda ng emergency supplies at inuming tubig.",
      avoidFloodProneRoutes: "Iwasan ang underpass, ilog, at bahain na daan.",
      followLocalAdvisories: "Sundin ang mga abiso mula sa lokal na awtoridad.",
      checkEvacuationOptions: "Alamin ang ligtas na daan at evacuation area.",

      // Severe Flooding
      stayIndoorsIfPossible: "Manatili muna sa loob ng bahay kung maaari.",
      avoidFloodwaters: "Iwasang lumusong o dumaan sa baha.",
      assistVulnerablePeople: "Tulungan ang bata, senior, at mga nangangailangan ng alalay.",
      prepareForEvacuation: "Maging handa sa agarang evacuation kung kinakailangan.",
      contactEmergencyServices: "Tumawag sa emergency services kung kailangan ng agarang tulong.",
      
      // Default Actions
      defaultDrinkWater:
        "Regular na uminom ng tubig, lalo na kung matagal nasa labas.",
      defaultWearComfortableClothing: "Magsuot ng magaan at komportableng damit.",
      defaultCheckUpdates: "Patuloy na mag-check ng mga update kung tumataas ang temperatura.",
      defaultStayCool: "Manatili sa lilim o malamig na lugar kung maaari.",
    },
  },
} satisfies Messages;
