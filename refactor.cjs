const fs = require('fs');
const lines = fs.readFileSync('app.py', 'utf8').split('\n');

const block1_start = 883;
const block1_end = 1280;

const block2_start = 3809;
const block2_end = 4104;

const block1 = lines.slice(block1_start, block1_end);
const unindented_block1 = block1.map(line => {
    if (line.startsWith('    ')) {
        return line.substring(4);
    }
    return line;
});

// Remove block 2
lines.splice(block2_start, block2_end - block2_start);

// Remove block 1
lines.splice(block1_start, block1_end - block1_start);

// Find insert index
let insert_idx = -1;
for (let i = 0; i < lines.length; i++) {
    if (lines[i].includes('def fetch_alpaca_history')) {
        for (let j = i + 1; j < lines.length; j++) {
            if (!lines[j].startsWith(' ') && lines[j].trim() !== '') {
                insert_idx = j;
                break;
            }
        }
        break;
    }
}

lines.splice(insert_idx, 0, ...unindented_block1);

fs.writeFileSync('app.py', lines.join('\n'));
console.log('Refactored app.py');
