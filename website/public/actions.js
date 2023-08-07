// If clicked on the screen, this function will be called
// 1. Get the x and y coordinates of the click
// 2. Find the corresponding cell in the crossword grid
// 3. If the cell is a black square, do nothing
// 4. If the cell is a white square, highlight the cell
Phaser.Scene.prototype.mouse_click = function(pointer) {
    let x = pointer.x;
    let y = pointer.y;

    console.log('Mouse clicked at x=' + x + ', y=' + y);

    // Get the cell coordinates
    let col = Math.floor(x / this.cellWidth);
    let row = Math.floor(y / this.cellHeight);

    console.log('Cell clicked row=' + row + ', col=' + col);

    let board_letter = this.board[row][col];

    if (board_letter == '*') {
        console.log('Black square clicked');
        return;
    }

    this.highlight_square.call(this, row, col);
}

Phaser.Scene.prototype.highlight_square = function(row, column) {
    // If the clicked cell is the same as the currently highlighted cell, toggle the direction
    if (this.highlighted && this.highlighted[0] === row && this.highlighted[1] === column) {
        this.curr_direction = this.curr_direction === 'across' ? 'down' : 'across'; // Toggle direction
        this.highlightedGraphics.clear(); // Clear previous graphics
    } else {
        // If something else is already highlighted, clear it
        if (this.highlightedGraphics) {
            this.highlightedGraphics.destroy(); // Destroy the previous graphics object
        }
        // Create a new graphics object for the highlight
        this.highlightedGraphics = this.add.graphics();
        this.highlighted = [row, column];
    }

    const graphics = this.highlightedGraphics;
    const alpha = 0.5; // Set opacity to 50%
    graphics.fillStyle(0x0000ff, alpha);
    graphics.fillRect(column * this.cellWidth, row * this.cellHeight, this.cellWidth, this.cellHeight);

    // Define a smaller size for the triangle and offset its position
    const triangleSize = Math.min(this.cellWidth, this.cellHeight) / 6; // Adjust the size of the triangle
    const offsetX = 3 * this.cellWidth / 20;
    const offsetY = 7 * this.cellHeight / 10;
    const centerX = column * this.cellWidth + offsetX;
    const centerY = row * this.cellHeight + offsetY;

    // Draw the triangle, depending on the direction
    graphics.fillStyle(0xff0000, alpha); // Setting the color of the triangle (change as needed)
    if (this.curr_direction === 'across') {
        // Draw a triangle pointing to the right
        graphics.fillTriangle(
            centerX, centerY - triangleSize / 2,
            centerX + triangleSize, centerY,
            centerX, centerY + triangleSize / 2
        );
    } else if (this.curr_direction === 'down') {
        // Draw a triangle pointing down
        graphics.fillTriangle(
            centerX - triangleSize / 2, centerY,
            centerX + triangleSize / 2, centerY,
            centerX, centerY + triangleSize
        );
    }
};

Phaser.Scene.prototype.key_press = function(event) {
    if (this.highlighted == null) {
        console.log('No cell highlighted');
        return;
    }

    const key_pressed = event.key.toLowerCase(); // Ensuring it's lowercase for consistent checks

    console.log('Key pressed: ' + key_pressed);

    // Get the current row and column of the highlighted cell
    let [row, col] = this.highlighted;

    if (key_pressed == "backspace"){ // Backspace
        const textObj = this.letter_grid[row][col];
        if (textObj) {
            textObj.setText(""); // Clear the letter
        }
        this.board[row][col] = '_'; // Clear the letter in the board as well

        // Move the highlight in the opposite direction
        if (this.curr_direction === 'across') {
            col--; // Move left
        } else if (this.curr_direction === 'down') {
            row--; // Move up
        }

        // Check if the new position is off the board or is a black square
        if (row < 0 || col < 0 || row >= this.board.length || col >= this.board[0].length || this.board[row][col] === '*') {
            this.highlighted = null; // Unhighlight
            if (this.highlightedGraphics) {
                this.highlightedGraphics.destroy(); // Destroy the graphics object
                this.highlightedGraphics = null;
            }
            return;
        }

        // Highlight the new cell
        this.highlight_square(row, col);
        return; // Exit early after handling backspace
    }
    else if (!(key_pressed.length === 1 && key_pressed.match(/[a-z]/))) {
        // If the key pressed is not a letter, do nothing
        console.log('Key pressed is not a letter');
        return;
    }

    // Get the letter in the highlighted cell
    const board_letter = this.board[row][col];

    // If the highlighted cell is a black square, do nothing
    if (board_letter == '*') {
        console.log('Black square highlighted');
        return;
    }

    // Update the board with the new letter
    this.board[row][col] = key_pressed;

    // Update the graphic/text for the highlighted cell
    const textObj = this.letter_grid[row][col];
    if (textObj) {
        textObj.setText(key_pressed.toUpperCase()); // Setting the letter in uppercase for display
    }

    // Move the highlight based on the current direction
    if (this.curr_direction === 'across') {
        col++; // Move right
    } else if (this.curr_direction === 'down') {
        row++; // Move down
    }

    // Check if the new position is off the board or is a black square
    if (row >= this.board.length || col >= this.board[0].length || this.board[row][col] === '*') {
        this.highlighted = null; // Unhighlight
        if (this.highlightedGraphics) {
            this.highlightedGraphics.destroy(); // Destroy the graphics object
            this.highlightedGraphics = null;
        }
        return;
    }

    // Highlight the new cell
    this.highlight_square(row, col);
};
