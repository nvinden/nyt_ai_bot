// Configuration object
var config = {
    type: Phaser.WEBGL,
    width: 800,
    height: 1000,
    backgroundColor: '#ffffff', // Set the background color here (hex color code)
    parent: 'crossword-container',
    scene: {
        preload: preload,
        create: create,
        update: update
    }
};

// Create the game with the configuration object
var game = new Phaser.Game(config);

// Preload assets (like images, audio, etc.)
// Loads all files
async function preload() {
    // Load javascript files
    console.log("preload");

    const json_file = "cross_data/complete_crosswords/puzzle_*LESSE***TSONE*_D2N7U_CHATGPT_gpt-4_MANU.json";
    let json_data;

    try {
      json_data = await loadJsonFile(json_file);
    } catch (error) {
      console.error('Error loading JSON file:', error);
    }

    this.complete_board = json_data.board;
    this.entries = json_data.entries;
    this.n_rows = json_data.n_rows;
    this.n_cols = json_data.n_cols;
    this.cellWidth = game.config.width / this.n_cols;
    this.cellHeight = (game.config.height - 200) / this.n_rows;
    this.highlighted = null;
    this.curr_direction = "across";

    let curr_board = [];

    // Iterate through the oldMatrix and modify entries
    for (let row = 0; row < this.complete_board.length; row++) {
      curr_board[row] = [];
      for (let col = 0; col < this.complete_board[row].length; col++) {
        // Check if the entry is an uppercase character
        if (this.complete_board[row][col].match(/[A-Z]/)) {
          curr_board[row][col] = '_'; // Replace with underscore
        } 
        else {
          curr_board[row][col] = this.complete_board[row][col]; // Keep the original value
        }
      }
    }

    this.board = curr_board;

    // Clue grids
    // Iterate through each clue in the provided list
    this.acrossClues = Array.from({ length: this.n_rows }, () => Array(this.n_cols).fill(null));
    this.downClues = Array.from({ length: this.n_rows }, () => Array(this.n_cols).fill(null));

    this.entries.forEach(clueObj => {
        if (clueObj.direction === 'across') {
            getAcrossClueString.call(this, clueObj);
        } else if (clueObj.direction === 'down') {
            getDownClueString.call(this, clueObj);
        }
    });

    this.cluePlacement = Array.from({ length: this.n_rows }, () => Array(this.n_cols).fill(null));
    this.entries.forEach(clueObj => {
        this.cluePlacement[clueObj.row][clueObj.col] = clueObj.clue_num;
    });

    console.log(this.board, this.entries, this.n_rows, this.n_cols);

    //await create();
}

// Create objects (called once after preload)
// 1. Creates datastructs for the crossword
// 2. Creates the crossword grid
// 3. Creates the clues
async function create() {
    const self = this;

    // Example: this.add.image(x, y, 'key');
    console.log("create");

    await delay(1000);

    console.log("wait done");

    this.drawGrid(this.board);

    // Add function to click on the screen
    this.input.on('pointerdown', function (pointer) {
        self.mouse_click.call(self, pointer); 
    });

    // Add function to press buttons
    this.input.keyboard.on('keydown', function (event) {
        self.key_press.call(self, event);
    });


}

// Update loop (called repeatedly after create)
function update() {
    // Example: game logic goes here
    if (this.board == undefined || this.highlighted == undefined) {
        console.log("waiting update");
        return;
    }

    console.log("update");

    this.drawClue.call(this);
}


// Helper Function
// Define the asynchronous function using async keyword
async function loadJsonFile(url) {
    try {
      const response = await fetch(url);
      if (!response.ok) {
        throw new Error('Network response was not ok');
      }
      const jsonData = await response.json();
      return jsonData;
    } catch (error) {
      return null;
    }
  }
  
  function isJSONValid(jsonString) {
    try {
      JSON.parse(jsonString);
      return true;
    } catch (error) {
      return false;
    }
  }

  async function isFileExists(filePath) {
    try {
      const response = await fetch(filePath, { method: 'HEAD' });
  
      // Check if the response status is in the 200 range (success) or 304 (not modified)
      return response.ok || response.status === 304;
    } catch (error) {
      // An error occurred, file might not exist or other network issues
      return false;
    }
  }
  function delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
  
function getAcrossClueString(clueObj) {
    let clueString = clueObj.clue;
    for (let col = clueObj.col; col < this.board[0].length; col++) {
        if (this.board[clueObj.row][col] === '*') break; // Stop if it reaches a black square
        this.acrossClues[clueObj.row][col] = [clueString, clueObj.clue_num]; // Add the clue number to the grid
    }
};

function getDownClueString(clueObj) {
    let clueString = clueObj.clue;
    for (let row = clueObj.row; row < this.board.length; row++) {
        if (this.board[row][clueObj.col] === '*') break; // Stop if it reaches a black square
        this.downClues[row][clueObj.col] = [clueString, clueObj.clue_num];
    }
};

