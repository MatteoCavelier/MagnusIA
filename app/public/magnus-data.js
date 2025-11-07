// --- 0. Configuration de Base ---

// Chemins d'icônes
const imagePaths = {
    'Rook': 'Ressources/rook.png',
    'Knight': 'Ressources/knight.png',
    'Bishop': 'Ressources/bishop.png',
    'Queen': 'Ressources/queen.png',
    'King': 'Ressources/king.png',
    'Pawn': 'Ressources/pawn.png'
};

// ** DONNÉES D'OUVERTURE (ECO/PGN) EXTRAITES DE GAMES.CSV ET CODES_ECO.PDF **
const openingsData = {
    // --- OUVERTURES DE FLANC (A) ---
    'A00': {
        name: "Ouverture Irrégulière",
        moves: "1. a3, 1. g4, 1. Nh3, etc."
    },
    'A04': {
        name: "Ouverture Réti",
        moves: "1. Nf3"
    },
    'A05': {
        name: "Ouverture Réti (divers)",
        moves: "1. Nf3 Nf6"
    },
    'A10': {
        name: "Ouverture Anglaise",
        moves: "1. c4"
    },
    'A11': {
        name: "Ouverture Anglaise, Système Caro-Kann",
        moves: "1. c4 c6"
    },

    // --- JEUX DU PION DAME HORS 1...d5 (A40-A99) ---
    'A41': {
        name: "Défense Indienne Ancienne",
        moves: "1. d4 d6"
    },
    'A45': {
        name: "Attaque Trompowsky ou Jeu du Pion Dame (divers)",
        moves: "1. d4 Nf6 2. Bg5"
    },
    'A46': {
        name: "Jeu du Pion Dame: Variante Torre",
        moves: "1. d4 Nf6 2. Nf3 e6 3. Bg5"
    },
    'A80': {
        name: "Défense Hollandaise",
        moves: "1. d4 f5"
    },

    // --- JEUX SEMI-OUVERTS (B) ---
    'B00': {
        name: "Défense Nimzowitsch ou Début du Pion Roi",
        moves: "1. e4 Nc6"
    },
    'B01': {
        name: "Défense Scandinave",
        moves: "1. e4 d5"
    },
    'B02': {
        name: "Défense Alekhine",
        moves: "1. e4 Nf6"
    },
    'B10': {
        name: "Défense Caro-Kann (divers)",
        moves: "1. e4 c6"
    },
    'B12': {
        name: "Défense Caro-Kann (Avance)",
        moves: "1. e4 c6 2. d4 d5 3. e5"
    },
    'B20': {
        name: "Défense Sicilienne (Anti-Siciliennes)",
        moves: "1. e4 c5"
    },
    'B21': {
        name: "Sicilienne, Attaque Grand Prix",
        moves: "1. e4 c5 2. f4"
    },
    // ... B30 à B99 (Variantes spécifiques de la Sicilienne)

    // --- JEUX OUVERTS (C) ---
    'C00': {
        name: "Défense Française",
        moves: "1. e4 e6"
    },
    'C10': {
        name: "Défense Française, Variante Rubinstein",
        moves: "1. e4 e6 2. d4 d5 3. Nd2 dxe4 4. Nxe4"
    },
    'C20': {
        name: "Début du Pion Roi (divers)",
        moves: "1. e4 e5 2. d3"
    },
    'C30': {
        name: "Gambit du Roi",
        moves: "1. e4 e5 2. f4"
    },
    'C41': {
        name: "Défense Philidor",
        moves: "1. e4 e5 2. Nf3 d6"
    },
    'C44': {
        name: "Partie Ponziani / Écossaise",
        moves: "1. e4 e5 2. Nf3 Nc6 3. c3 / 3. d4"
    },
    'C50': {
        name: "Partie Italienne (divers)",
        moves: "1. e4 e5 2. Nf3 Nc6 3. Bc4"
    },
    'C70': {
        name: "Ouverture Espagnole (divers)",
        moves: "1. e4 e5 2. Nf3 Nc6 3. Bb5"
    },

    // --- JEUX FERMÉS (D) ---
    'D00': {
        name: "Jeu du Pion Dame (divers)",
        moves: "1. d4 d5 2. Bg5 / 2. e3"
    },
    'D10': {
        name: "Défense Slave",
        moves: "1. d4 d5 2. c4 c6"
    },
    'D50': {
        name: "Gambit Dame Refusé",
        moves: "1. d4 d5 2. c4 e6"
    },
    'D70': {
        name: "Défense Néo-Grünfeld",
        moves: "1. d4 Nf6 2. c4 g6 3. f3 d5"
    },
    'D90': {
        name: "Défense Grünfeld",
        moves: "1. d4 Nf6 2. c4 g6 3. Nc3 d5"
    },

    // --- DÉFENSES INDIENNES (E) ---
    'E00': {
        name: "Jeu du Pion Dame (divers) / Ouverture Catalane",
        moves: "1. d4 Nf6 2. c4 e6 3. g3"
    },
    'E10': {
        name: "Défense Bogo-Indienne / Gambit Blumenfeld",
        moves: "1. d4 Nf6 2. c4 e6 3. Nf3"
    },
    'E60': {
        name: "Défense Est-Indienne",
        moves: "1. d4 Nf6 2. c4 g6"
    },
};

console.log(openingsData);

// Position FEN de départ
const startingFEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

// Mappage FEN à Pièce
const fenPieceMap = {
    'r': { path: imagePaths.Rook, color: 'black' }, 'n': { path: imagePaths.Knight, color: 'black' },
    'b': { path: imagePaths.Bishop, color: 'black' }, 'q': { path: imagePaths.Queen, color: 'black' },
    'k': { path: imagePaths.King, color: 'black' }, 'p': { path: imagePaths.Pawn, color: 'black' },
    'R': { path: imagePaths.Rook, color: 'white' }, 'N': { path: imagePaths.Knight, color: 'white' },
    'B': { path: imagePaths.Bishop, color: 'white' }, 'Q': { path: imagePaths.Queen, color: 'white' },
    'K': { path: imagePaths.King, color: 'white' }, 'P': { path: imagePaths.Pawn, color: 'white' }
};