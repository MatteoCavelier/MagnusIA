// ** INITIALISATION CHESS.JS **
const chess = new Chess();
let currentThemeName = 'glacial'; // Thème par défaut (doit correspondre à la sélection initiale dans le HTML)
let lastPredictedWinner = null; // Stocke le dernier gagnant prédit pour mise à jour dynamique


// --- Fonction de Prédiction avec Validation ---
async function predictOutcome() {
    const API_URL = "http://127.0.0.1:1234/predict/noduration-noturn-novstatus-None-True";

    // Récupération des éléments d'entrée
    const eloWhiteInput = document.getElementById('input-white');
    const eloBlackInput = document.getElementById('input-black');
    const openingCodeInput = document.getElementById('input-ouverture-search');
    const numMovesInput = document.getElementById('input-num-move');
    const incrementCodeInput = document.getElementById('input-increment-code');

    const resultContainer = document.getElementById('result-container');
    const predictionText = document.getElementById('prediction-text');

    // Récupération et conversion des valeurs
    const eloWhite = parseInt(eloWhiteInput.value);
    const eloBlack = parseInt(eloBlackInput.value);
    const openingCode = openingCodeInput.value.trim();
    const num_moves = parseInt(numMovesInput.value);
    const increment_code = incrementCodeInput.value.trim();

    // --- 1. VALIDATION DES CHAMPS ---
    const errors = [];

    // Fonction utilitaire pour gérer l'affichage des erreurs sur les inputs
    const validateInput = (inputElement, condition, errorMessage) => {
        if (condition) {
            errors.push(errorMessage);
            inputElement.classList.add('border-4', 'border-red-500');
        } else {
            inputElement.classList.remove('border-4', 'border-red-500');
        }
    };

    validateInput(eloWhiteInput, isNaN(eloWhite) || eloWhite <= 0, "L'Elo du joueur BLANC est requis et doit être positif.");
    validateInput(eloBlackInput, isNaN(eloBlack) || eloBlack <= 0, "L'Elo du joueur NOIR est requis et doit être positif.");
    validateInput(numMovesInput, isNaN(num_moves) || num_moves <= 0, "Le nombre de coups estimés pour l'ouverture est requis et doit être supérieur à zéro.");
    validateInput(openingCodeInput, openingCode === "", "Le Code d'Ouverture (ECO) est requis.");
    validateInput(incrementCodeInput, increment_code === "", "La Cadence des coups est requise (Ex: 10+5).");


    // Si des erreurs existent, affichez-les et arrêtez l'exécution
    if (errors.length > 0) {
        console.error("Erreurs de validation:", errors);

        predictionText.innerHTML = `<strong class="text-red-500">Veuillez corriger les champs suivants:</strong><br>${errors.join('<br>')}`;
        resultContainer.classList.remove("hidden");
        return; // ARRÊT : N'envoie pas la requête Fetch
    }

    // Affichage en cours de prédiction
    predictionText.textContent = "Prédiction en cours...";
    resultContainer.classList.remove("hidden");

    // --- 2. ENVOI DE LA REQUÊTE ---
    const payload = {
        "increment_code": increment_code,
        "white_rating": eloWhite,
        "black_rating": eloBlack,
        "opening_eco": openingCode,
        "opening_ply": num_moves
    };

    console.log("requete envoyée : ", payload);

    try {
        const response = await fetch(API_URL, {
            method: "POST",
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(payload),
        });

        console.log("Objet Response brut : ", response);

        if (!response.ok) {
            const errorDetails = await response.text();
            console.error(`Erreur HTTP: ${response.status} ${response.statusText}`, errorDetails);

            predictionText.textContent = `Erreur du serveur (${response.status}): Veuillez vérifier l'API.`;
            resultContainer.classList.remove("hidden");
            return;
        }

        const data = await response.json();

        console.log("Données de prédiction reçues :", data);

        const winner = data.prediction_label[0].toUpperCase(); // "WHITE" ou "BLACK"

        // Stocker le gagnant pour les changements de thème futurs
        lastPredictedWinner = winner;

        // Mettre à jour l'affichage avec le thème actuel
        updatePredictionDisplay(winner);

        resultContainer.classList.remove("hidden")

    } catch (e) {
        console.error("Erreur lors de la lecture du JSON ou de la requête Fetch:", e);
        predictionText.textContent = "Erreur: Impossible de contacter le serveur de prédiction (URL ou CORS).";
        resultContainer.classList.remove("hidden");
    }
}


// --- Fonction pour mettre à jour l'affichage de la prédiction ---
function updatePredictionDisplay(winner) {
    const predictionText = document.getElementById('prediction-text');
    const theme = themes[currentThemeName];
    let colorClass = '';
    let winnerText = '';

    // Déterminer la couleur et le texte en français (harmonisation thématique)
    if (winner === 'WHITE') {
        colorClass = theme.strongTextColorWhite;
        winnerText = 'BLANC';
    } else if (winner === 'BLACK') {
        colorClass = theme.strongTextColorBlack;
        winnerText = 'NOIR';
    } else {
        // Cas de match nul ou résultat inattendu
        colorClass = theme.plainText;
        winnerText = winner;
    }

    // Nettoyer toutes les anciennes classes de couleur
    Object.values(themes).forEach(t => {
        const strongEl = predictionText.querySelector('strong');
        if (strongEl) {
            strongEl.classList.remove(t.strongTextColorWhite, t.strongTextColorBlack);
        }
    });

    // Mettre à jour l'affichage
    if (winner === 'WHITE' || winner === 'BLACK') {
        predictionText.innerHTML = `
            Gagnant prévu : 
            <strong class="font-bold ${colorClass}">
                ${winnerText}
            </strong>
        `;
    } else {
        predictionText.textContent = `Gagnant prévu : ${winnerText}`;
    }
}


// --- Logique d'Affichage FEN sur l'Échiquier (inchangée) ---
function updateChessboardFromFEN(fen, themeData) {
    const board = document.getElementById('chessboard');
    board.innerHTML = '';
    const positionPart = fen.split(' ')[0];
    let positionIndex = 0;
    let cellIndex = 0;

    while (cellIndex < 64 && positionIndex < positionPart.length) {
        const row = Math.floor(cellIndex / 8);
        const col = cellIndex % 8;
        const isDark = (row + col) % 2 === 1;
        const cell = document.createElement('div');

        cell.className = isDark ? themeData.darkSquare : themeData.lightSquare;
        cell.classList.add('transition-colors', 'duration-500', 'flex', 'justify-center', 'items-center');

        let currentFENChar = positionPart[positionIndex];

        if (/\d/.test(currentFENChar)) {
            const emptySquares = parseInt(currentFENChar, 10);
            for (let i = 0; i < emptySquares; i++) {
                const emptyCell = document.createElement('div');
                const isDarkEmpty = (Math.floor((cellIndex + i) / 8) + (cellIndex + i) % 8) % 2 === 1;
                emptyCell.className = isDarkEmpty ? themeData.darkSquare : themeData.lightSquare;
                emptyCell.classList.add('transition-colors', 'duration-500', 'flex', 'justify-center', 'items-center');
                board.appendChild(emptyCell);
            }
            cellIndex += emptySquares;
            positionIndex++;
        } else if (currentFENChar === '/') {
            positionIndex++;
        } else if (fenPieceMap[currentFENChar]) {
            const pieceData = fenPieceMap[currentFENChar];
            const pieceFilterClass = (pieceData.color === 'white')
                ? themeData.pieceFilterClassWhite
                : themeData.pieceFilterClassBlack;

            const pieceElement = document.createElement('img');
            pieceElement.src = pieceData.path;
            pieceElement.alt = "Pièce d'échecs";
            pieceElement.className = `piece-icon ${pieceFilterClass}`;
            cell.appendChild(pieceElement);
            board.appendChild(cell);

            cellIndex++;
            positionIndex++;
        } else {
            positionIndex++;
        }
    }
}


/**
 * Gère la recherche et la conversion ECO (coups) -> FEN -> Affichage. (inchangée)
 */
function updateBoardForOpening(inputValue) {
    const input = inputValue.toUpperCase().trim();
    const theme = themes[currentThemeName];
    const openingNameEl = document.getElementById('opening-name');

    let targetOpening = null;

    for (const code in openingsData) {
        if (code === input || openingsData[code].name.toUpperCase().includes(input)) {
            targetOpening = openingsData[code];
            break;
        }
    }

    if (targetOpening) {
        chess.reset();

        const moves = targetOpening.moves.split(/\s+/).filter(m => m && !m.includes('.'));

        try {
            for (const move of moves) {
                chess.move(move);
            }

            const currentFEN = chess.fen();
            updateChessboardFromFEN(currentFEN, theme);
            openingNameEl.textContent = `${targetOpening.name} (${targetOpening.code})`;
        } catch (e) {
            console.error("Erreur lors de l'application des coups PGN:", e);
            chess.reset();
            updateChessboardFromFEN(startingFEN, theme);
            openingNameEl.textContent = 'Erreur PGN / Position de départ';
        }


    } else {
        chess.reset();
        updateChessboardFromFEN(startingFEN, theme);
        openingNameEl.textContent = 'Position de départ';
    }
}


// --- 2. Génération de l'Échiquier (Initialisation) (inchangée) ---
function generateChessboard(themeData) {
    chess.reset();
    updateChessboardFromFEN(startingFEN, themeData);

    document.getElementById('opening-name').textContent = 'Position de départ';
}

// --- Initialisation Datalist (Uniquement avec les codes ECO) (inchangée) ---
function initializeOpeningsDatalist() {
    const datalist = document.getElementById('opening-codes');
    datalist.innerHTML = '';
    for (const code in openingsData) {
        const optionCode = document.createElement('option');
        optionCode.value = code;
        optionCode.textContent = openingsData[code].name;
        datalist.appendChild(optionCode);
    }
}

// --- Fonction d'Application du Thème (Modifiée pour inclure prediction-text) ---
function applyTheme(themeName) {
    const theme = themes[themeName];
    if (!theme) return;

    currentThemeName = themeName;

    const whiteStrong = document.getElementById('strong-white');
    const blackStrong = document.getElementById('strong-black');
    const resultContainer = document.getElementById('result-container');
    const openingNameEl = document.getElementById('opening-name');
    const selectEl = document.getElementById('theme-select');

    // Nettoyage et application des classes fortes/résultats
    Object.values(themes).forEach(t => {
        whiteStrong.classList.remove(t.strongTextColorWhite);
        blackStrong.classList.remove(t.strongTextColorBlack);
        openingNameEl.classList.remove(t.plainText);
        if (t.result) resultContainer.classList.remove(...t.result.split(' '), 'border');
        selectEl.classList.remove(t.focusRing.replace('focus:', ''));
    });

    whiteStrong.classList.add(theme.strongTextColorWhite);
    blackStrong.classList.add(theme.strongTextColorBlack);
    resultContainer.classList.add(...theme.result.split(' '));
    resultContainer.classList.add('border');
    openingNameEl.classList.add(theme.plainText);
    selectEl.classList.add(theme.focusRing.replace('focus:', ''));


    const elements = [
        {id: 'body', classes: theme.body, reset: true},
        {id: 'main-card', classes: theme.card, reset: true},
        {id: 'title-h1', classes: theme.title, reset: true},
        {id: 'title-h2', classes: theme.title, reset: true},
        {id: 'title-h3', classes: theme.title, reset: true},
        {id: 'label-white', classes: theme.plainText, reset: true},
        {id: 'label-black', classes: theme.plainText, reset: true},
        {id: 'label-increment-code', classes: theme.plainText, reset: true},
        {id: 'label-ouverture-search', classes: theme.plainText, reset: true},
        {id: 'label-num-move', classes: theme.plainText, reset: true},
        {id: 'label-board-state', classes: theme.plainText, reset: true},
        {id: 'prediction-text', classes: theme.plainText, reset: true}, // AJOUTÉ : Pour colorer le texte statique "Gagnant prévu : "
        {id: 'divider', classes: theme.divider, reset: true},
        {id: 'divider-2', classes: theme.divider, reset: true},
        {id: 'predict-button', classes: theme.button, reset: true},
        {
            id: 'input-white',
            classes: `${theme.inputBase} focus:outline-none focus:ring-2 ${theme.focusRing} focus:ring-opacity-75`,
            reset: true
        },
        {
            id: 'input-black',
            classes: `${theme.inputBase} focus:outline-none focus:ring-2 ${theme.focusRing} focus:ring-opacity-75`,
            reset: true
        },
        {
            id: 'input-increment-code',
            classes: `${theme.inputBase} focus:outline-none focus:ring-2 ${theme.focusRing} focus:ring-opacity-75`,
            reset: true
        },
        {
            id: 'input-ouverture-search',
            classes: `${theme.inputBase} focus:outline-none focus:ring-2 ${theme.focusRing} focus:ring-opacity-75`,
            reset: true
        },
        {
            id: 'input-num-move',
            classes: `${theme.inputBase} focus:outline-none focus:ring-2 ${theme.focusRing} focus:ring-opacity-75`,
            reset: true
        },
    ];

    elements.forEach(elData => {
        const el = document.getElementById(elData.id);
        if (el) {
            Object.values(themes).forEach(t => {
                let classesToClear = [t.body, t.card, t.title, t.plainText, t.divider, t.button, t.inputBase, t.focusRing, 'focus:outline-none', 'focus:ring-2', 'focus:ring-opacity-75'];
                classesToClear.forEach(classString => {
                    el.classList.remove(...classString.split(' '));
                });
            });

            el.classList.add(...elData.classes.split(' '));

            if (el.tagName === 'INPUT' || el.tagName === 'BUTTON' || el.tagName === 'HR') {
                el.classList.add('transition-colors', 'duration-500');
            }
        }
    });

    const currentOpeningInput = document.getElementById('input-ouverture-search').value;
    if (currentOpeningInput) {
        updateBoardForOpening(currentOpeningInput);
    } else {
        generateChessboard(theme);
    }

    // Mettre à jour l'affichage de la prédiction si elle existe
    if (lastPredictedWinner) {
        updatePredictionDisplay(lastPredictedWinner);
    }
}

// --- Initialisation au Chargement de la Page (inchangée) ---
document.addEventListener('DOMContentLoaded', () => {
    initializeOpeningsDatalist();
    const initialTheme = document.getElementById('theme-select').value;
    if (initialTheme) {
        applyTheme(initialTheme);
    }
});