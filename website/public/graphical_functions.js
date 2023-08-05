Phaser.Scene.prototype.drawGrid = function(board) {
    // Create a graphics object to draw the grid and letters
    const graphics = this.add.graphics();

    // Set the line style for the grid lines
    graphics.lineStyle(1, 0x000000, 1);

    // Cell width and height
    const cellWidth = this.game.config.width / board[0].length;
    const cellHeight = this.game.config.height / board.length;

    // Draw vertical lines to create the columns and display letters in each cell
    for (let col = 0; col <= board[0].length; col++) {
        const x = col * cellWidth;
        graphics.moveTo(x, 0);
        graphics.lineTo(x, this.game.config.height);

        for (let row = 0; row < board.length; row++) {
            const y = row * cellHeight;
            const letter = board[row][col];

            if (letter == "*") {
                const_square = this.add.rectangle(x + cellWidth / 2, y + cellHeight / 2, cellWidth, cellHeight, 0x000000);
                const_square.setOrigin(0.5, 0.5);
            }
            else if (letter == "_") {
                continue;
            }
            else {
                // Display the letter in the cell
                const text = this.add.text(x + cellWidth / 2, y + cellHeight / 2, letter, {
                    font: '24px Arial',
                    fill: '#000000',
                    align: 'center'
                });
                text.setOrigin(0.5, 0.5);
            }
        }
    }

    // Draw horizontal lines to create the rows
    for (let row = 0; row <= board.length; row++) {
        const y = row * cellHeight;
        graphics.moveTo(0, y);
        graphics.lineTo(this.game.config.width, y);
    }

    // Render the grid on the screen
    graphics.strokePath();
    this.graphics = graphics;
};
