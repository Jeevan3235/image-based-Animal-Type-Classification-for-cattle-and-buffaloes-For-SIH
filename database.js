// Comprehensive Buffalo and Cattle Breed Database
// Accurate breed information for classification and identification

const BREED_DATABASE = {
    buffalo: {
        murrah: {
            name: "Murrah",
            origin: "India (Haryana)",
            characteristics: {
                color: "Jet black",
                bodyType: "Heavy, compact body",
                horns: "Short, tightly curled",
                size: "Large",
                weight: "Male: 500-600kg, Female: 400-500kg"
            },
            colorProfile: {
                black: 0.85,
                gray: 0.10,
                brown: 0.03,
                white: 0.02
            },
            milkProduction: "High (1500-2500 liters/lactation)",
            temperament: "Docile, good for dairy",
            distribution: "India, Pakistan, Bangladesh, Nepal",
            temperatureTolerance: "10-35°C",
            climateAdaptation: "Well adapted to tropical and subtropical climates",
            feedEfficiency: "Good, requires 12-15kg dry matter per day",
            diseaseResistance: "Moderate resistance to tropical diseases"
        },
        jaffarabadi: {
            name: "Jaffarabadi",
            origin: "India (Gujarat)",
            characteristics: {
                color: "Dark gray to black",
                bodyType: "Massive, heavy build",
                horns: "Long, curved downward",
                size: "Very large",
                weight: "Male: 600-800kg, Female: 500-600kg"
            },
            colorProfile: {
                black: 0.70,
                gray: 0.25,
                brown: 0.03,
                white: 0.02
            },
            milkProduction: "Very high (2000-3000 liters/lactation)",
            temperament: "Calm, excellent dairy breed",
            distribution: "India (Gujarat), Pakistan",
            temperatureTolerance: "12-38°C",
            climateAdaptation: "Highly adapted to hot and humid coastal regions",
            feedEfficiency: "Excellent, requires 15-18kg dry matter per day",
            diseaseResistance: "Good resistance to local diseases and parasites"
        },
        surti: {
            name: "Surti",
            origin: "India (Gujarat)",
            characteristics: {
                color: "Light gray to dark gray",
                bodyType: "Medium, well-proportioned",
                horns: "Medium length, curved",
                size: "Medium",
                weight: "Male: 400-500kg, Female: 350-450kg"
            },
            colorProfile: {
                black: 0.20,
                gray: 0.70,
                brown: 0.05,
                white: 0.05
            },
            milkProduction: "Good (1200-1800 liters/lactation)",
            temperament: "Gentle, easy to handle",
            distribution: "India (Gujarat, Maharashtra)",
            temperatureTolerance: "15-40°C",
            climateAdaptation: "Well adapted to semi-arid regions",
            feedEfficiency: "Good, requires 10-12kg dry matter per day",
            diseaseResistance: "Moderate to good resistance to local diseases"
        },
        mehsana: {
            name: "Mehsana",
            origin: "India (Gujarat)",
            characteristics: {
                color: "Black with white markings",
                bodyType: "Medium to large",
                horns: "Medium, curved",
                size: "Medium-large",
                weight: "Male: 450-550kg, Female: 400-500kg"
            },
            colorProfile: {
                black: 0.60,
                gray: 0.20,
                brown: 0.05,
                white: 0.15
            },
            milkProduction: "High (1500-2200 liters/lactation)",
            temperament: "Docile, good milkers",
            distribution: "India (Gujarat)"
        },
        nili_ravi: {
            name: "Nili-Ravi",
            origin: "Pakistan (Punjab)",
            characteristics: {
                color: "Black with white markings",
                bodyType: "Heavy, well-built",
                horns: "Short, tightly curled",
                size: "Large",
                weight: "Male: 500-600kg, Female: 450-550kg"
            },
            colorProfile: {
                black: 0.70,
                gray: 0.15,
                brown: 0.05,
                white: 0.10
            },
            milkProduction: "Very high (1800-2500 liters/lactation)",
            temperament: "Calm, excellent dairy",
            distribution: "Pakistan, India (Punjab)"
        },
        bhadawari: {
            name: "Bhadawari",
            origin: "India (Uttar Pradesh)",
            characteristics: {
                color: "Copper brown to dark brown",
                bodyType: "Medium, compact",
                horns: "Medium length",
                size: "Medium",
                weight: "Male: 400-500kg, Female: 350-450kg"
            },
            colorProfile: {
                black: 0.10,
                gray: 0.20,
                brown: 0.65,
                white: 0.05
            },
            milkProduction: "Good (1000-1500 liters/lactation)",
            temperament: "Hardy, good for draught",
            distribution: "India (Uttar Pradesh, Madhya Pradesh)"
        },
        kalahandi: {
            name: "Kalahandi",
            origin: "India (Odisha)",
            characteristics: {
                color: "Dark gray to black",
                bodyType: "Medium, sturdy",
                horns: "Medium length",
                size: "Medium",
                weight: "Male: 400-500kg, Female: 350-450kg"
            },
            colorProfile: {
                black: 0.50,
                gray: 0.40,
                brown: 0.08,
                white: 0.02
            },
            milkProduction: "Moderate (800-1200 liters/lactation)",
            temperament: "Hardy, good for work",
            distribution: "India (Odisha)"
        },
        tedo: {
            name: "Tedo",
            origin: "India (Assam)",
            characteristics: {
                color: "Dark gray to black",
                bodyType: "Medium, well-built",
                horns: "Medium length",
                size: "Medium",
                weight: "Male: 400-500kg, Female: 350-450kg"
            },
            colorProfile: {
                black: 0.60,
                gray: 0.30,
                brown: 0.08,
                white: 0.02
            },
            milkProduction: "Good (1000-1500 liters/lactation)",
            temperament: "Adaptable, good milkers",
            distribution: "India (Assam, Northeast)"
        },
        bhadawari: {
            name: "Bhadawari",
            origin: "India (Uttar Pradesh)",
            characteristics: {
                color: "Copper brown to dark brown",
                bodyType: "Medium, compact",
                horns: "Medium length",
                size: "Medium",
                weight: "Male: 400-500kg, Female: 350-450kg"
            },
            colorProfile: {
                black: 0.10,
                gray: 0.20,
                brown: 0.65,
                white: 0.05
            },
            milkProduction: "Good (1000-1500 liters/lactation)",
            temperament: "Hardy, good for draught",
            distribution: "India (Uttar Pradesh, Madhya Pradesh)"
        },
        banni: {
            name: "Banni",
            origin: "India (Gujarat)",
            characteristics: {
                color: "Black to dark gray",
                bodyType: "Medium, sturdy",
                horns: "Medium length, curved",
                size: "Medium",
                weight: "Male: 450-550kg, Female: 400-500kg"
            },
            colorProfile: {
                black: 0.70,
                gray: 0.25,
                brown: 0.03,
                white: 0.02
            },
            milkProduction: "Good (1200-1800 liters/lactation)",
            temperament: "Hardy, good milkers",
            distribution: "India (Gujarat, Kutch region)"
        },
        chilika: {
            name: "Chilika",
            origin: "India (Odisha)",
            characteristics: {
                color: "Dark gray to black",
                bodyType: "Medium, well-built",
                horns: "Medium length",
                size: "Medium",
                weight: "Male: 400-500kg, Female: 350-450kg"
            },
            colorProfile: {
                black: 0.60,
                gray: 0.35,
                brown: 0.03,
                white: 0.02
            },
            milkProduction: "Good (1000-1500 liters/lactation)",
            temperament: "Adaptable, good for work",
            distribution: "India (Odisha, Chilika lake region)"
        },
        godawari: {
            name: "Godawari",
            origin: "India (Maharashtra)",
            characteristics: {
                color: "Dark gray to black",
                bodyType: "Medium, compact",
                horns: "Medium length",
                size: "Medium",
                weight: "Male: 400-500kg, Female: 350-450kg"
            },
            colorProfile: {
                black: 0.65,
                gray: 0.30,
                brown: 0.03,
                white: 0.02
            },
            milkProduction: "Good (1000-1500 liters/lactation)",
            temperament: "Hardy, good for work",
            distribution: "India (Maharashtra, Godavari region)"
        },
        jerangi: {
            name: "Jerangi",
            origin: "India (Odisha)",
            characteristics: {
                color: "Dark gray to black",
                bodyType: "Medium, sturdy",
                horns: "Medium length",
                size: "Medium",
                weight: "Male: 400-500kg, Female: 350-450kg"
            },
            colorProfile: {
                black: 0.55,
                gray: 0.40,
                brown: 0.03,
                white: 0.02
            },
            milkProduction: "Moderate (800-1200 liters/lactation)",
            temperament: "Hardy, good for work",
            distribution: "India (Odisha, Jerangi region)"
        },
        kujang: {
            name: "Kujang",
            origin: "India (Odisha)",
            characteristics: {
                color: "Dark gray to black",
                bodyType: "Medium, compact",
                horns: "Medium length",
                size: "Medium",
                weight: "Male: 400-500kg, Female: 350-450kg"
            },
            colorProfile: {
                black: 0.60,
                gray: 0.35,
                brown: 0.03,
                white: 0.02
            },
            milkProduction: "Moderate (800-1200 liters/lactation)",
            temperament: "Hardy, good for work",
            distribution: "India (Odisha, Kujang region)"
        },
        manda: {
            name: "Manda",
            origin: "India (Odisha)",
            characteristics: {
                color: "Dark gray to black",
                bodyType: "Medium, sturdy",
                horns: "Medium length",
                size: "Medium",
                weight: "Male: 400-500kg, Female: 350-450kg"
            },
            colorProfile: {
                black: 0.65,
                gray: 0.30,
                brown: 0.03,
                white: 0.02
            },
            milkProduction: "Moderate (800-1200 liters/lactation)",
            temperament: "Hardy, good for work",
            distribution: "India (Odisha, Manda region)"
        },
        marathwada: {
            name: "Marathwada",
            origin: "India (Maharashtra)",
            characteristics: {
                color: "Dark gray to black",
                bodyType: "Medium, well-built",
                horns: "Medium length",
                size: "Medium",
                weight: "Male: 400-500kg, Female: 350-450kg"
            },
            colorProfile: {
                black: 0.60,
                gray: 0.35,
                brown: 0.03,
                white: 0.02
            },
            milkProduction: "Good (1000-1500 liters/lactation)",
            temperament: "Hardy, good milkers",
            distribution: "India (Maharashtra, Marathwada region)"
        },
        nagpuri: {
            name: "Nagpuri",
            origin: "India (Maharashtra)",
            characteristics: {
                color: "Black to dark gray",
                bodyType: "Medium to large",
                horns: "Medium length, curved",
                size: "Medium-large",
                weight: "Male: 450-550kg, Female: 400-500kg"
            },
            colorProfile: {
                black: 0.70,
                gray: 0.25,
                brown: 0.03,
                white: 0.02
            },
            milkProduction: "Good (1200-1800 liters/lactation)",
            temperament: "Docile, good milkers",
            distribution: "India (Maharashtra, Nagpur region)"
        },
        pandharpuri: {
            name: "Pandharpuri",
            origin: "India (Maharashtra)",
            characteristics: {
                color: "Black to dark gray",
                bodyType: "Medium, well-built",
                horns: "Medium length, curved",
                size: "Medium",
                weight: "Male: 400-500kg, Female: 350-450kg"
            },
            colorProfile: {
                black: 0.65,
                gray: 0.30,
                brown: 0.03,
                white: 0.02
            },
            milkProduction: "Good (1000-1500 liters/lactation)",
            temperament: "Docile, good milkers",
            distribution: "India (Maharashtra, Pandharpur region)"
        },
        paralakhemundi: {
            name: "Paralakhemundi",
            origin: "India (Odisha)",
            characteristics: {
                color: "Dark gray to black",
                bodyType: "Medium, sturdy",
                horns: "Medium length",
                size: "Medium",
                weight: "Male: 400-500kg, Female: 350-450kg"
            },
            colorProfile: {
                black: 0.60,
                gray: 0.35,
                brown: 0.03,
                white: 0.02
            },
            milkProduction: "Moderate (800-1200 liters/lactation)",
            temperament: "Hardy, good for work",
            distribution: "India (Odisha, Paralakhemundi region)"
        },
        sambalpuri: {
            name: "Sambalpuri",
            origin: "India (Odisha)",
            characteristics: {
                color: "Dark gray to black",
                bodyType: "Medium, compact",
                horns: "Medium length",
                size: "Medium",
                weight: "Male: 400-500kg, Female: 350-450kg"
            },
            colorProfile: {
                black: 0.65,
                gray: 0.30,
                brown: 0.03,
                white: 0.02
            },
            milkProduction: "Moderate (800-1200 liters/lactation)",
            temperament: "Hardy, good for work",
            distribution: "India (Odisha, Sambalpur region)"
        },
        south_kanara: {
            name: "South Kanara",
            origin: "India (Karnataka)",
            characteristics: {
                color: "Dark gray to black",
                bodyType: "Medium, well-built",
                horns: "Medium length",
                size: "Medium",
                weight: "Male: 400-500kg, Female: 350-450kg"
            },
            colorProfile: {
                black: 0.60,
                gray: 0.35,
                brown: 0.03,
                white: 0.02
            },
            milkProduction: "Good (1000-1500 liters/lactation)",
            temperament: "Adaptable, good milkers",
            distribution: "India (Karnataka, South Kanara region)"
        },
        swamp_buffalo: {
            name: "Swamp Buffalo",
            origin: "Southeast Asia",
            characteristics: {
                color: "Dark gray to black",
                bodyType: "Large, heavy",
                horns: "Long, curved",
                size: "Large",
                weight: "Male: 500-700kg, Female: 400-600kg"
            },
            colorProfile: {
                black: 0.70,
                gray: 0.25,
                brown: 0.03,
                white: 0.02
            },
            milkProduction: "Moderate (800-1200 liters/lactation)",
            temperament: "Calm, good for work",
            distribution: "Southeast Asia, India (Northeast)"
        },
        tarai: {
            name: "Tarai",
            origin: "India (Uttarakhand)",
            characteristics: {
                color: "Dark gray to black",
                bodyType: "Medium, sturdy",
                horns: "Medium length",
                size: "Medium",
                weight: "Male: 400-500kg, Female: 350-450kg"
            },
            colorProfile: {
                black: 0.60,
                gray: 0.35,
                brown: 0.03,
                white: 0.02
            },
            milkProduction: "Good (1000-1500 liters/lactation)",
            temperament: "Hardy, good for work",
            distribution: "India (Uttarakhand, Tarai region)"
        },
        toda: {
            name: "Toda",
            origin: "India (Tamil Nadu)",
            characteristics: {
                color: "Dark gray to black",
                bodyType: "Medium, compact",
                horns: "Medium length",
                size: "Medium",
                weight: "Male: 400-500kg, Female: 350-450kg"
            },
            colorProfile: {
                black: 0.65,
                gray: 0.30,
                brown: 0.03,
                white: 0.02
            },
            milkProduction: "Good (1000-1500 liters/lactation)",
            temperament: "Gentle, good milkers",
            distribution: "India (Tamil Nadu, Nilgiri hills)"
        }
    },
    cattle: {
        holstein_friesian: {
            name: "Holstein Friesian",
            origin: "Netherlands/Germany",
            characteristics: {
                color: "Black and white spotted",
                bodyType: "Large, angular",
                horns: "Polled or horned",
                size: "Very large",
                weight: "Male: 700-900kg, Female: 500-700kg"
            },
            colorProfile: {
                black: 0.40,
                gray: 0.05,
                brown: 0.05,
                white: 0.50
            },
            milkProduction: "Very high (6000-8000 liters/lactation)",
            temperament: "Docile, high milk producers",
            distribution: "Worldwide (dairy industry)"
        },
        jersey: {
            name: "Jersey",
            origin: "Jersey Island (UK)",
            characteristics: {
                color: "Light brown to dark brown",
                bodyType: "Small, refined",
                horns: "Small, curved",
                size: "Small",
                weight: "Male: 400-500kg, Female: 300-400kg"
            },
            colorProfile: {
                black: 0.05,
                gray: 0.05,
                brown: 0.85,
                white: 0.05
            },
            milkProduction: "High (4000-6000 liters/lactation)",
            temperament: "Gentle, efficient milkers",
            distribution: "Worldwide (dairy industry)"
        },
        sahiwal: {
            name: "Sahiwal",
            origin: "Pakistan/India",
            characteristics: {
                color: "Red to dark red",
                bodyType: "Medium to large",
                horns: "Medium length",
                size: "Medium-large",
                weight: "Male: 500-600kg, Female: 400-500kg"
            },
            colorProfile: {
                black: 0.05,
                gray: 0.05,
                brown: 0.85,
                white: 0.05
            },
            milkProduction: "Good (2000-3000 liters/lactation)",
            temperament: "Hardy, heat tolerant",
            distribution: "India, Pakistan, Bangladesh"
        },
        gir: {
            name: "Gir",
            origin: "India (Gujarat)",
            characteristics: {
                color: "Red to dark red with white spots",
                bodyType: "Large, well-built",
                horns: "Long, curved",
                size: "Large",
                weight: "Male: 500-600kg, Female: 400-500kg"
            },
            colorProfile: {
                black: 0.05,
                gray: 0.05,
                brown: 0.70,
                white: 0.20
            },
            milkProduction: "Good (2000-3000 liters/lactation)",
            temperament: "Docile, good milkers",
            distribution: "India (Gujarat, Rajasthan)"
        },
        red_sindhi: {
            name: "Red Sindhi",
            origin: "Pakistan (Sindh)",
            characteristics: {
                color: "Deep red",
                bodyType: "Medium, compact",
                horns: "Medium length",
                size: "Medium",
                weight: "Male: 400-500kg, Female: 300-400kg"
            },
            colorProfile: {
                black: 0.05,
                gray: 0.05,
                brown: 0.85,
                white: 0.05
            },
            milkProduction: "Good (1500-2500 liters/lactation)",
            temperament: "Hardy, heat tolerant",
            distribution: "Pakistan, India, Bangladesh"
        },
        tharparkar: {
            name: "Tharparkar",
            origin: "India (Rajasthan)",
            characteristics: {
                color: "White to light gray",
                bodyType: "Medium to large",
                horns: "Medium length",
                size: "Medium-large",
                weight: "Male: 450-550kg, Female: 350-450kg"
            },
            colorProfile: {
                black: 0.05,
                gray: 0.20,
                brown: 0.10,
                white: 0.65
            },
            milkProduction: "Good (1500-2500 liters/lactation)",
            temperament: "Hardy, drought resistant",
            distribution: "India (Rajasthan, Gujarat)"
        },
        kankrej: {
            name: "Kankrej",
            origin: "India (Gujarat)",
            characteristics: {
                color: "Silver gray to dark gray",
                bodyType: "Large, powerful",
                horns: "Long, curved",
                size: "Large",
                weight: "Male: 500-600kg, Female: 400-500kg"
            },
            colorProfile: {
                black: 0.10,
                gray: 0.80,
                brown: 0.05,
                white: 0.05
            },
            milkProduction: "Good (1500-2500 liters/lactation)",
            temperament: "Strong, good for work",
            distribution: "India (Gujarat, Rajasthan)"
        },
        hariana: {
            name: "Hariana",
            origin: "India (Haryana)",
            characteristics: {
                color: "White to light gray",
                bodyType: "Medium to large",
                horns: "Medium length",
                size: "Medium-large",
                weight: "Male: 450-550kg, Female: 350-450kg"
            },
            colorProfile: {
                black: 0.05,
                gray: 0.15,
                brown: 0.10,
                white: 0.70
            },
            milkProduction: "Good (1500-2500 liters/lactation)",
            temperament: "Hardy, good milkers",
            distribution: "India (Haryana, Punjab, UP)"
        },
        ongole: {
            name: "Ongole",
            origin: "India (Andhra Pradesh)",
            characteristics: {
                color: "White to light gray",
                bodyType: "Large, muscular",
                horns: "Medium length",
                size: "Large",
                weight: "Male: 500-600kg, Female: 400-500kg"
            },
            colorProfile: {
                black: 0.05,
                gray: 0.20,
                brown: 0.10,
                white: 0.65
            },
            milkProduction: "Good (1500-2500 liters/lactation)",
            temperament: "Strong, good for work",
            distribution: "India (Andhra Pradesh, Telangana)"
        },
        deoni: {
            name: "Deoni",
            origin: "India (Maharashtra)",
            characteristics: {
                color: "Black and white spotted",
                bodyType: "Medium to large",
                horns: "Medium length",
                size: "Medium-large",
                weight: "Male: 450-550kg, Female: 350-450kg"
            },
            colorProfile: {
                black: 0.50,
                gray: 0.05,
                brown: 0.05,
                white: 0.40
            },
            milkProduction: "Good (1500-2500 liters/lactation)",
            temperament: "Hardy, dual purpose",
            distribution: "India (Maharashtra, Karnataka)"
        },
        amritmahal: {
            name: "Amritmahal",
            origin: "India (Karnataka)",
            characteristics: {
                color: "Gray to dark gray",
                bodyType: "Medium, compact",
                horns: "Medium length",
                size: "Medium",
                weight: "Male: 400-500kg, Female: 300-400kg"
            },
            colorProfile: {
                black: 0.10,
                gray: 0.80,
                brown: 0.05,
                white: 0.05
            },
            milkProduction: "Moderate (1000-1500 liters/lactation)",
            temperament: "Hardy, good for work",
            distribution: "India (Karnataka)"
        },
        hallikar: {
            name: "Hallikar",
            origin: "India (Karnataka)",
            characteristics: {
                color: "Gray to dark gray",
                bodyType: "Medium, compact",
                horns: "Medium length",
                size: "Medium",
                weight: "Male: 400-500kg, Female: 300-400kg"
            },
            colorProfile: {
                black: 0.10,
                gray: 0.80,
                brown: 0.05,
                white: 0.05
            },
            milkProduction: "Moderate (1000-1500 liters/lactation)",
            temperament: "Hardy, good for work",
            distribution: "India (Karnataka)"
        },
        khillari: {
            name: "Khillari",
            origin: "India (Maharashtra)",
            characteristics: {
                color: "White to light gray",
                bodyType: "Medium, compact",
                horns: "Medium length",
                size: "Medium",
                weight: "Male: 400-500kg, Female: 300-400kg"
            },
            colorProfile: {
                black: 0.05,
                gray: 0.20,
                brown: 0.10,
                white: 0.65
            },
            milkProduction: "Moderate (1000-1500 liters/lactation)",
            temperament: "Hardy, good for work",
            distribution: "India (Maharashtra, Karnataka)"
        },
        malvi: {
            name: "Malvi",
            origin: "India (Madhya Pradesh)",
            characteristics: {
                color: "White to light gray",
                bodyType: "Medium, compact",
                horns: "Medium length",
                size: "Medium",
                weight: "Male: 400-500kg, Female: 300-400kg"
            },
            colorProfile: {
                black: 0.05,
                gray: 0.20,
                brown: 0.10,
                white: 0.65
            },
            milkProduction: "Moderate (1000-1500 liters/lactation)",
            temperament: "Hardy, good for work",
            distribution: "India (Madhya Pradesh, Rajasthan)"
        },
        nimari: {
            name: "Nimari",
            origin: "India (Madhya Pradesh)",
            characteristics: {
                color: "Red and white spotted",
                bodyType: "Medium, compact",
                horns: "Medium length",
                size: "Medium",
                weight: "Male: 400-500kg, Female: 300-400kg"
            },
            colorProfile: {
                black: 0.05,
                gray: 0.05,
                brown: 0.60,
                white: 0.30
            },
            milkProduction: "Moderate (1000-1500 liters/lactation)",
            temperament: "Hardy, good for work",
            distribution: "India (Madhya Pradesh)"
        },
        punganur: {
            name: "Punganur",
            origin: "India (Andhra Pradesh)",
            characteristics: {
                color: "White to light gray",
                bodyType: "Small, compact",
                horns: "Small, curved",
                size: "Small",
                weight: "Male: 200-300kg, Female: 150-250kg"
            },
            colorProfile: {
                black: 0.05,
                gray: 0.20,
                brown: 0.10,
                white: 0.65
            },
            milkProduction: "Low (500-1000 liters/lactation)",
            temperament: "Hardy, drought resistant",
            distribution: "India (Andhra Pradesh)"
        },
        vechur: {
            name: "Vechur",
            origin: "India (Kerala)",
            characteristics: {
                color: "Light brown to dark brown",
                bodyType: "Very small, compact",
                horns: "Small, curved",
                size: "Very small",
                weight: "Male: 150-200kg, Female: 100-150kg"
            },
            colorProfile: {
                black: 0.05,
                gray: 0.05,
                brown: 0.85,
                white: 0.05
            },
            milkProduction: "Low (500-1000 liters/lactation)",
            temperament: "Gentle, easy to handle",
            distribution: "India (Kerala)"
        }
    }
};

// Breed classification functions
function findBestMatchingBreed(type, colorProfile) {
    const breeds = BREED_DATABASE[type.toLowerCase()];
    if (!breeds) return { breed: 'Unknown', confidence: 0, details: null };

    let bestMatch = { breed: 'Unknown', confidence: 0, details: null };
    let bestScore = 0;

    for (const [breedKey, breedData] of Object.entries(breeds)) {
        const score = calculateBreedMatchScore(breedData.colorProfile, colorProfile);
        if (score > bestScore) {
            bestScore = score;
            bestMatch = {
                breed: breedData.name,
                confidence: Math.round(score * 100),
                details: breedData
            };
        }
    }

    // Only return a breed if confidence is above threshold
    if (bestMatch.confidence < 50) {
        return { breed: 'Unknown', confidence: 0, details: null };
    }

    return bestMatch;
}

function calculateBreedMatchScore(breedProfile, imageProfile) {
    let score = 0;
    const colors = ['black', 'gray', 'brown', 'white'];
    
    for (const color of colors) {
        const breedValue = breedProfile[color] || 0;
        const imageValue = imageProfile[color] || 0;
        // Calculate similarity (1 - absolute difference)
        const similarity = 1 - Math.abs(breedValue - imageValue);
        score += similarity * (breedValue + imageValue) / 2; // Weight by expected presence
    }
    
    return Math.min(score / colors.length, 1); // Normalize to 0-1
}

function getBreedInformation(breedName, type) {
    const breeds = BREED_DATABASE[type.toLowerCase()];
    if (!breeds) return null;

    for (const [breedKey, breedData] of Object.entries(breeds)) {
        if (breedData.name.toLowerCase() === breedName.toLowerCase()) {
            return breedData;
        }
    }
    return null;
}

function getAllBreeds(type) {
    return BREED_DATABASE[type.toLowerCase()] || {};
}

function getBreedStatistics() {
    const stats = {
        buffalo: Object.keys(BREED_DATABASE.buffalo).length,
        cattle: Object.keys(BREED_DATABASE.cattle).length,
        total: Object.keys(BREED_DATABASE.buffalo).length + Object.keys(BREED_DATABASE.cattle).length
    };
    return stats;
}

// Export for use in other files
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        BREED_DATABASE,
        findBestMatchingBreed,
        calculateBreedMatchScore,
        getBreedInformation,
        getAllBreeds,
        getBreedStatistics
    };
}
