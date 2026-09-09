/**
 * This script is intended to inform of the potentials for how an explainable
 * system may fail.
 *
 * Command line parameters are:
 *
 *   -k <kb_path> The knowledgebase folder containing with the saved confusion
 *         matrix in the file cm.json.
 *
 *   -p <class_id> The id of the class predicted.
 *
 * Output to the console will include the false discovery rate (FDR) and the
 * descending list of the classes with the highest likelihood of causing
 * confusion given the prediction.
 */

const fs = require('fs');
const { program } = require('commander');

program
  .requiredOption('-k, --knowledgebase <path>', 'Must indicate the path to the kb')
  .requiredOption('-p, --prediction  <number>', 'integer argument', parseInt)
program.parse();

const options = program.opts();

let cm = JSON.parse(fs.readFileSync(options.knowledgebase + "/cm.json"));
let cm_length = cm.length

let tp = cm[options.prediction][options.prediction];

let fp = 0;

let mistakes = [];

for (let row = 0; row < cm.length; row++) {
    if (row != options.prediction) {
        fp += cm[row][options.prediction];
        if (cm[row][options.prediction] > 0) {
            mistakes.push({row: row, count: cm[row][options.prediction]});
        }
    }
}

console.log("FDR", fp / (fp + tp) );

mistakes.sort((a, b) => (a.count > b.count ? -1 : 1));

for (let m of mistakes) {
    m["pct"] = m.count / (tp + fp)
}

console.log(mistakes);
