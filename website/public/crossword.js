// Configuration object
var config = {
    type: Phaser.WEBGL,
    width: 800,
    height: 600,
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

    const json_file = "data/complete_crosswords/puzzle_*LESSE***TSONE*_D2N7U_CHATGPT_gpt-4_MANU.json";
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

    console.log(this.board, this.entries, this.n_rows, this.n_cols);

    //await create();
}

// Create objects (called once after preload)
// 1. Creates datastructs for the crossword
// 2. Creates the crossword grid
// 3. Creates the clues
async function create() {
    // Example: this.add.image(x, y, 'key');
    console.log("create");

    await delay(1000);

    console.log("wait done");

    this.drawGrid(this.board);

    const mouse_click = this.mouse_click;

    // Add function to click on the screen
    this.input.on('pointerdown', function (pointer) {
      mouse_click(pointer);
    });
}

// Update loop (called repeatedly after create)
function update() {
    // Example: game logic goes here
    if (this.board == undefined) {
        console.log("waiting update");
        return;
    }

    console.log("update");

    //console.log(this.board, this.entries, this.n_rows, this.n_cols);
}


// Graphical function
function drawGrid(board) {
  // Create a graphics object to draw the grid and letters
  const graphics = this.add.graphics();

  // Set the line style for the grid lines
  graphics.lineStyle(1, 0x000000, 1);

  // Cell width and height
  const cellWidth = game.config.width / board[0].length;
  const cellHeight = game.config.height / board.length;

  // Draw vertical lines to create the columns and display letters in each cell
  for (let col = 0; col <= board[0].length; col++) {
      const x = col * cellWidth;
      graphics.moveTo(x, 0);
      graphics.lineTo(x, game.config.height);

      for (let row = 0; row < board.length; row++) {
          const y = row * cellHeight;
          const letter = board[row][col];

          // Display the letter in the cell
          const text = this.add.text(x + cellWidth / 2, y + cellHeight / 2, letter, {
              font: '24px Arial',
              fill: '#000000',
              align: 'center'
          });
          text.setOrigin(0.5, 0.5);
      }
  }

  // Draw horizontal lines to create the rows
  for (let row = 0; row <= board.length; row++) {
      const y = row * cellHeight;
      graphics.moveTo(0, y);
      graphics.lineTo(game.config.width, y);
  }

  // Render the grid on the screen
  graphics.strokePath();
  this.graphics = graphics;
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
  
  