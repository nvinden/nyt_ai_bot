Phaser.Scene.prototype.drawGrid = function(board) {
    // Create a graphics object to draw the grid and letters
    const graphics = this.add.graphics();

    // Set the line style for the grid lines
    graphics.lineStyle(1, 0x000000, 1);

    // Cell width and height
    const cellWidth = this.cellWidth;
    const cellHeight = this.cellHeight;

    let letter_grid = [];

    // Initialize letter_grid with arrays for each row
    for (let i = 0; i < board.length; i++) {
        letter_grid[i] = [];
    }

    // Draw vertical lines to create the columns and display letters in each cell
    for (let col = 0; col < board[0].length; col++) {
        const x = col * cellWidth;
        graphics.moveTo(x, 0);
        graphics.lineTo(x, this.game.config.height);

        for (let row = 0; row < board.length; row++) {
            const y = row * cellHeight;
            const letter = board[row][col];

            // Check if there's a clue number in this.cluePlacement for the current cell
            const clueNumber = this.cluePlacement[row][col];
            if (clueNumber !== null) {
                const clueText = this.add.text(x + 5, y + 5, clueNumber.toString(), {
                    font: '16px Arial',
                    fill: '#000000'
                });
            }

            if (letter == "*") {
                const_square = this.add.rectangle(x + cellWidth / 2, y + cellHeight / 2, cellWidth, cellHeight, 0x000000);
                const_square.setOrigin(0.5, 0.5);
                letter_grid[row][col] = null; // Store null in the designated row and column for "*"
            } else if (letter == "_") {
                // Display an empty text object in the cell for "_"
                const text = this.add.text(x + 9 * cellWidth / 16, y + 5 * cellHeight / 8, "", {
                    font: '24px Arial',
                    fill: '#000000',
                    align: 'center'
                });
                text.setOrigin(0.5, 0.5);
                letter_grid[row][col] = text; // Store the empty text object in the designated row and column
            } else {
                // Display the letter in the cell
                const text = this.add.text(x + 9 * cellWidth / 16, y + 5 * cellHeight / 8, letter, {
                    font: '24px Arial',
                    fill: '#000000',
                    align: 'center'
                });
                text.setOrigin(0.5, 0.5);
                letter_grid[row][col] = text; // Store the letter's graphics object in the designated row and column
            }
        }
    }

    // Draw horizontal lines to create the rows
    for (let row = 0; row <= board.length; row++) {
        const y = row * cellHeight;
        graphics.moveTo(0, y);
        graphics.lineTo(this.game.config.width, y);
    }

    // Set the line style for the grid lines
    graphics.lineStyle(1, 0x000000, 1);

    // Render the grid on the screen
    graphics.strokePath();
    this.graphics = graphics;
    this.letter_grid = letter_grid; // Store the letter_grid array in the scene

    // Create a separate graphics object to draw the white box with a black outline
    const boxGraphics = this.add.graphics();
    const whiteBoxHeight = 200;
    const whiteBoxY = this.game.config.height - whiteBoxHeight;
    boxGraphics.fillStyle(0xffffff, 1); // Set the fill color to white
    boxGraphics.fillRect(0, whiteBoxY, this.game.config.width, whiteBoxHeight); // Draw the filled rectangle
    boxGraphics.lineStyle(1, 0x000000, 1); // Set the line style for the outline
    boxGraphics.strokeRect(0, whiteBoxY, this.game.config.width, whiteBoxHeight); // Draw the outline 

    this.clue_text = this.add.text(10, whiteBoxY + 10, "Welcome to the crossword", {
        font: '24px Arial',
        fill: '#000000',
        align: 'left',
        wordWrap: { width: this.game.config.width - 20 } // Wrap the text at the width of the game minus some padding
    });
};

Phaser.Scene.prototype.drawClue = function() {
    //console.log("drawClue");

    if (this.highlighted == null) {
        this.clue_text.setText(''); // Clear the clue text
        return;
    }

    // Get the highlighed cell's row and column
    const row = this.highlighted[0];
    const col = this.highlighted[1];
    let clue;
    let clue_num;


    // Get the clue for the highlighted cell
    if (this.curr_direction == "across"){
        clue = this.acrossClues[row][col][0]
        clue_num = this.acrossClues[row][col][1];
    }
    else{
        clue = this.downClues[row][col][0]
        clue_num = this.downClues[row][col][1];
    }

    // Display the clue
    let clue_text = "Clue " + clue_num + " " + this.curr_direction + ":\n" + clue;
    this.clue_text.setText(clue_text);



}
