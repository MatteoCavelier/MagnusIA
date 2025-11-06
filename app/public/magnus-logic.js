// Les variables et dépendances globales doivent être déclarées ici si elles sont utilisées par plusieurs fonctions.
// Elles dépendent de la présence de magnus-data.js et magnus-themes.js.

// ** INITIALISATION CHESS.JS **
const chess = new Chess();
let currentThemeName = 'glacial'; // Thème par défaut (doit correspondre à la sélection initiale dans le HTML)


// --- Fonction de Prédiction (Exemple) ---
function predictOutcome() {
    const eloWhite = parseInt(document.getElementById('input-white').value);
    const eloBlack = parseInt(document.getElementById('input-black').value);
    const openingCode = document.getElementById('input-ouverture-search').value;
    const num_moves = document.getElementById('input-num-move').value;
    const increment_code = document.getElementById('input-increment-code').value;
    const turns = document.getElementById('input-turns').value;

    const resultContainer = document.getElementById('result-container');
    const predictionText = document.getElementById('prediction-text');

}


// --- Logique d'Affichage FEN sur l'Échiquier ---
function updateChessboardFromFEN(fen, themeData) {
    const board = document.getElementById('chessboard');
    board.innerHTML = '';
    const positionPart = fen.split(' ')[0]; // Ne prend que la disposition du plateau
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

        if (/\d/.test(currentFENChar)) { // Cases vides
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
        }
        else if (currentFENChar === '/') { // Fin de rangée
            positionIndex++;
        }
        else if (fenPieceMap[currentFENChar]) { // Pièce
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
 * Gère la recherche et la conversion ECO (coups) -> FEN -> Affichage.
 * La recherche se fait par code ECO OU par nom d'ouverture.
 */
function updateBoardForOpening(inputValue) {
    const input = inputValue.toUpperCase().trim();
    const theme = themes[currentThemeName];
    const openingNameEl = document.getElementById('opening-name');

    let targetOpening = null;

    // Chercher par code ECO ou par nom
    for (const code in openingsData) {
        if (code === input || openingsData[code].name.toUpperCase().includes(input)) {
            targetOpening = openingsData[code];
            break;
        }
    }

    if (targetOpening) {
        // Le cas où l'utilisateur entre le NOM, on met à jour l'input pour afficher le CODE
        if (targetOpening.name.toUpperCase() === input && document.getElementById('input-ouverture-search').value !== targetOpening.code) {
             // Non nécessaire ici car le datalist n'affiche que les codes, mais bonne pratique.
        }

        chess.reset();

        // Nettoyer et charger les coups
        const moves = targetOpening.moves.split(/\s+/).filter(m => m && !m.includes('.'));

        try {
            for (const move of moves) {
                chess.move(move); // chess.js s'occupe de la validation et de la mise à jour de l'état
            }

            const currentFEN = chess.fen();
            updateChessboardFromFEN(currentFEN, theme);
            openingNameEl.textContent = `${targetOpening.name} (${Object.keys(openingsData).find(key => openingsData[key] === targetOpening)})`;
        } catch (e) {
            console.error("Erreur lors de l'application des coups PGN:", e);
            chess.reset();
            updateChessboardFromFEN(startingFEN, theme);
            openingNameEl.textContent = 'Erreur PGN / Position de départ';
        }


    } else {
        // Réinitialiser
        chess.reset();
        updateChessboardFromFEN(startingFEN, theme);
        openingNameEl.textContent = 'Position de départ';
    }
}


// --- 2. Génération de l'Échiquier (Initialisation) ---
function generateChessboard(themeData) {
    // Initialise l'échiquier avec la position de départ
    chess.reset();
    updateChessboardFromFEN(startingFEN, themeData);

    document.getElementById('opening-name').textContent = 'Position de départ';
    // document.getElementById('input-ouverture-search').value = ''; // On ne réinitialise pas la valeur de l'input ici
}

// --- Initialisation Datalist (Uniquement avec les codes ECO) ---
function initializeOpeningsDatalist() {
    const datalist = document.getElementById('opening-codes');
    datalist.innerHTML = '';
    for (const code in openingsData) {
        const optionCode = document.createElement('option');
        optionCode.value = code; // La valeur de l'option est le CODE
        optionCode.textContent = openingsData[code].name; // On peut mettre le nom dans textContent
        datalist.appendChild(optionCode);
    }
}

// --- Fonction d'Application du Thème ---
function applyTheme(themeName) {
    const theme = themes[themeName];
    if (!theme) return;

    currentThemeName = themeName;
    // ... (Logique d'application des classes de thème inchangée) ...
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
    // On retire l'ajout de la bordure ici pour ne pas la nettoyer ensuite
    resultContainer.classList.add(...theme.result.split(' '));
    resultContainer.classList.add('border'); // Remettre border
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
        {id: 'label-turns', classes: theme.plainText, reset: true},
        {id: 'label-increment-code', classes: theme.plainText, reset: true},
        {id: 'label-ouverture-search', classes: theme.plainText, reset: true},
        {id: 'label-num-move', classes: theme.plainText, reset: true},
        {id: 'label-board-state', classes: theme.plainText, reset: true},
        {id: 'divider', classes: theme.divider, reset: true},
        {id: 'divider-2', classes: theme.divider, reset: true},
        {id: 'predict-button', classes: theme.button, reset: true},
        {
            id: 'input-white', classes: `${theme.inputBase} focus:outline-none focus:ring-2 ${theme.focusRing} focus:ring-opacity-75`, reset: true
        },
        {
            id: 'input-black', classes: `${theme.inputBase} focus:outline-none focus:ring-2 ${theme.focusRing} focus:ring-opacity-75`, reset: true
        },
        {
            id: 'input-increment-code', classes: `${theme.inputBase} focus:outline-none focus:ring-2 ${theme.focusRing} focus:ring-opacity-75`, reset: true
        },
        {
            id: 'input-turns', classes: `${theme.inputBase} focus:outline-none focus:ring-2 ${theme.focusRing} focus:ring-opacity-75`, reset: true
        },
        {
            id: 'input-ouverture-search', classes: `${theme.inputBase} focus:outline-none focus:ring-2 ${theme.focusRing} focus:ring-opacity-75`, reset: true
        },
        {
            id: 'input-num-move', classes: `${theme.inputBase} focus:outline-none focus:ring-2 ${theme.focusRing} focus:ring-opacity-75`, reset: true
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

    // Régénérer l'échiquier (et maintenir l'état si un code est déjà entré)
    const currentOpeningInput = document.getElementById('input-ouverture-search').value;
    if (currentOpeningInput) {
        updateBoardForOpening(currentOpeningInput);
    } else {
        generateChessboard(theme);
    }
}

// --- Initialisation au Chargement de la Page ---
document.addEventListener('DOMContentLoaded', () => {
    initializeOpeningsDatalist();
    // Assurez-vous que le thème initial est appliqué
    const initialTheme = document.getElementById('theme-select').value;
    if (initialTheme) {
        applyTheme(initialTheme);
    }
});