// Mock bird data for dummy backend responses
// Replace with real API calls when backend is connected

export const MOCK_BIRDS = {
  'westan': {
    commonName: 'Western Tanager',
    scientificName: 'Piranga ludoviciana',
    confidence: 0.87,
    family: 'Cardinalidae',
    habitat: 'Coniferous and mixed forests',
    diet: 'Insects, berries, and fruit',
    funFacts: [
      'Males sport a brilliant red head with yellow and black body — one of North America\'s most colorful birds.',
      'Despite their tropical appearance, Western Tanagers breed across western North America and migrate to Central America.',
      'Their song is a burry, robin-like phrase often described as "pit-er-ick".',
    ],
    range: {
      center: [45.5, -116.0],
      description: 'Western North America, from Alaska south to Mexico during breeding season.',
    },
    topPredictions: [
      { name: 'Western Tanager', confidence: 0.87 },
      { name: 'American Robin',  confidence: 0.08 },
      { name: 'Varied Thrush',   confidence: 0.05 },
    ],
    wikiImage: 'https://upload.wikimedia.org/wikipedia/commons/thumb/3/thirty/Piranga_ludoviciana_-_Western_Tanager.jpg/480px-Piranga_ludoviciana_-_Western_Tanager.jpg',
    color: '#c9a84c',
  },
  'norcar': {
    commonName: 'Northern Cardinal',
    scientificName: 'Cardinalis cardinalis',
    confidence: 0.92,
    family: 'Cardinalidae',
    habitat: 'Woodlands, gardens, shrublands',
    diet: 'Seeds, fruit, and insects',
    funFacts: [
      'The male\'s brilliant red plumage comes from carotenoid pigments in its diet.',
      'Unlike most songbirds, female Northern Cardinals also sing — a rare trait among North American birds.',
      'They are non-migratory and often visit backyard feeders year-round.',
    ],
    range: {
      center: [37.0, -85.0],
      description: 'Eastern and central North America, from southern Canada to Mexico.',
    },
    topPredictions: [
      { name: 'Northern Cardinal', confidence: 0.92 },
      { name: 'House Finch',       confidence: 0.05 },
      { name: 'Purple Finch',      confidence: 0.03 },
    ],
    wikiImage: 'https://upload.wikimedia.org/wikipedia/commons/thumb/a/a7/Cardinalis_cardinalis_-_20070909.jpg/480px-Cardinalis_cardinalis_-_20070909.jpg',
    color: '#c4714a',
  },
  'baleag': {
    commonName: 'Bald Eagle',
    scientificName: 'Haliaeetus leucocephalus',
    confidence: 0.78,
    family: 'Accipitridae',
    habitat: 'Coasts, rivers, large lakes',
    diet: 'Fish, waterfowl, small mammals',
    funFacts: [
      'Bald Eagles can spot a fish from nearly two miles away thanks to eyesight four times sharper than humans.',
      'Their iconic white head and tail don\'t appear until they reach maturity at age 4–5.',
      'A Bald Eagle\'s nest, called an eyrie, can weigh over a ton after years of additions.',
    ],
    range: {
      center: [55.0, -105.0],
      description: 'Across North America, especially near large bodies of open water.',
    },
    topPredictions: [
      { name: 'Bald Eagle',     confidence: 0.78 },
      { name: 'Osprey',         confidence: 0.14 },
      { name: 'Golden Eagle',   confidence: 0.08 },
    ],
    wikiImage: 'https://upload.wikimedia.org/wikipedia/commons/thumb/1/1e/Bald_Eagle_Portrait.jpg/480px-Bald_Eagle_Portrait.jpg',
    color: '#7a9e7e',
  },
}

export const SAMPLE_AUDIOS = [
  {
    id: 'westan',
    label: 'Western Tanager',
    description: 'Coniferous forest, Pacific Northwest',
    file: '/samples/western_tanager.ogg',
    emoji: '🟡',
  },
  {
    id: 'norcar',
    label: 'Northern Cardinal',
    description: 'Backyard garden, Eastern US',
    file: '/samples/northern_cardinal.ogg',
    emoji: '🔴',
  },
  {
    id: 'baleag',
    label: 'Bald Eagle',
    description: 'Lakeside at dawn, Pacific Northwest',
    file: '/samples/bald_eagle.ogg',
    emoji: '🟤',
  },
]

/**
 * Simulate a backend call with a fixed delay.
 * Replace this with a real fetch() to your FastAPI backend.
 *
 * @param {string} birdId - Key into MOCK_BIRDS
 * @returns {Promise<object>} - Mock prediction result
 */
export async function mockPredict(birdId = 'norcar') {
  await new Promise(r => setTimeout(r, 2200))
  return MOCK_BIRDS[birdId] ?? MOCK_BIRDS['norcar']
}
