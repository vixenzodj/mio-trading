import fs from 'fs';
let code = fs.readFileSync('app.py', 'utf8');
code = code.replace(/return long_sig\.fillna\(False\), short_sig\.fillna\(False\)/g, 'return long_sig.reindex(df.index, fill_value=False), short_sig.reindex(df.index, fill_value=False)');
fs.writeFileSync('app.py', code);
console.log('Replaced successfully');
