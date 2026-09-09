const fs = require('fs');
const path = require('path');
const { program } = require('commander');

program
  .requiredOption('-k, --knowledgebase <path>', 'Must indicate the path to the kb')
  .requiredOption('-e, --effectiveness_metric <metric_name>', 'Must name a metric to use for effectiveness')
  .option('-u, --unexplainable', 'Use an unexplainable component to increase performance');

program.parse();

const options = program.opts();

console.log("Running with options:", options);
// output the contribution metrics
get_contribution(undefined, undefined, options.effectiveness_metric);

const transform_names = [
        'crossing', 'endpoint', 'fill', 'skel-fill', 'skel', 'thresh',
        'line', 'ellipse', 'circle', 'ellipse-circle', 'chull', 'raw', 'corner', 'pca'
];

/**
 * Tells if the transform should be processed
 * @param {*} t_name - the transform name
 * @param {*} unexplainable - 
 */
function process_transform(t_name, unexplainable) {
    if (t_name == "thresh" || t_name == "skel-fill" || (!unexplainable && t_name == "raw")) {
        // exclude threshold, skeleton-fill, and the raw if we are not doing a mixed_explainability
        return false
    }
    return true;
}

let max_label = 0
let min_label = 0xffffffff


let kb = {
    transform_names: [],
    // hashmap storage for per transform statistics
    //   each element is an array of stats for a class
    stats: {},
    // contents of the stats json file
    metrics: {},
    // test label array for confirming all match
    //  each element is a label
    labels: undefined,
    // hashmap storage for test predictions per transform
    //   each element is an array of predictions
    test_preds: {},
    // Storage for the test probability
    test_prob: {}
}


for (let t_name of transform_names) {
    console.log(`=======${t_name}=======`);
    kb.transform_names.push(t_name);
    let auc = JSON.parse(fs.readFileSync(path.join(options.knowledgebase, `${t_name}_auc.json`)));
    let train_preds = JSON.parse(fs.readFileSync(path.join(options.knowledgebase, `${t_name}.train.json`)));
    let train_labels = JSON.parse(fs.readFileSync(path.join(options.knowledgebase, `${t_name}.train-labels.json`)));
    let test_preds = JSON.parse(fs.readFileSync(path.join(options.knowledgebase, `${t_name}.test.json`)));
    let test_labels = JSON.parse(fs.readFileSync(path.join(options.knowledgebase, `${t_name}.test-labels.json`)));
    let trans_stats = JSON.parse(fs.readFileSync(path.join(options.knowledgebase, `${t_name}-stat.json`)))
    let test_proba = JSON.parse(fs.readFileSync(path.join(options.knowledgebase, `${t_name}.test-proba.json`)));

    if (!kb.labels) {
        console.log("setting test labels");
        kb.labels = test_labels;
    } else {
        // check for all label arrays to be set and all equal
        if (test_labels.length === kb.labels.length && test_labels.every((element, index) => element === kb.labels[index])) {
            console.log("labels match");
        } else {
            // the test labels must match otherwise not comparing like data across transforms
            console.log("Error: labels do not match");
            process.exit(-1);
        }
    }

    kb.stats[t_name] = get_confusion_matrix(train_labels, train_preds, auc);
    //kb.stats[t_name]["auc"] = auc;
    kb.test_preds[t_name] = test_preds;
    kb.metrics[t_name] = trans_stats;
    kb.test_prob[t_name] = test_proba;
}

// build up the predictions and statistical metrics
let predictions = [];
let label_index = 0;
for (let label of kb.labels) {
    let poll = {
        label: label,
        index: label_index,
        votes: []
    };
    for (let t_name of transform_names) {
        if ( process_transform(t_name, options.unexplainable)) {

            let trans_prediction = kb.test_preds[t_name][label_index];

            poll.votes.push({
                trans_name: t_name,
                class: trans_prediction,
                stats: kb.stats[t_name]['stats'][trans_prediction],
                contribution: get_contribution(kb.stats[t_name]['stats'][trans_prediction], kb.metrics[t_name]['classes'][trans_prediction], options.effectiveness_metric),
                explainability: kb.metrics[t_name]['explainability'],
                ie_prob: kb.test_prob[t_name][label_index][trans_prediction] // output of the inference engine
                // kb.metrics[t_name]['classes'][trans_prediction]
            });

            // vote from the transform
            // console.log("label:", label, "transform:", t_name, "prediction:", trans_prediction);

            // stats for transform and class digit zero
            //console.log("zero:", kb.stats[t_name]['stats'][0]);
        }
    }
    predictions.push(poll);
    label_index++;
}

let correct = 0;
let second_choice = 0;
let count = 0;
// determine the results by combining predictions and weighting
let results = [];
let final_predictions = [];
let final_labels = [];
for (let p of predictions) {

    // r represents each test element we are gathering "a vote tally" for
    let r = {
        label: p.label,
        index: p.index,
        predictions: [],
        sum_weights: 0,
        vote_tally: [],
    };

    final_labels.push(p.label);

    let label_ix = min_label;
    while (label_ix <= max_label) {

        // represents the votes for a class
        let c = {
            class: label_ix,
            value: 0.0,
            attributions: [],
            exp_sum: 0
        }

        if (p && p.votes && p.votes.length > 0) {
            for (let v of p.votes) {
                if (v.class == label_ix) {
                    c.value += v.contribution * v.ie_prob;
                    // contribution as the inference engine output and contribution
                    c.attributions.push({name: v.trans_name, value: v.contribution * v.ie_prob, exp: v.explainability});
                    c.exp_sum += v.contribution * v.explainability;
                }
            };
        } else {
            console.log("invalid p.votes:", p);
        }

        if (c.value > 0.0) {
            c.attributions.sort((a, b) => {
                return b.value - a.value;
            })
            r.vote_tally.push(c);
        }

        label_ix++;
    }

    r.vote_tally.sort((a, b) => {
        return b.value - a.value;
    });

    for (let t of r.vote_tally) {
        r.sum_weights += t.value;
    }
    let action = "selected";
    // TODO: should probability (str_prob and t.probability) variables be changed to confidence?
    for (let t of r.vote_tally) {
        t["probability"] = t.value / r.sum_weights;
        t["exp"] = t.exp_sum / t.value;
        let str_prob = t.probability.toPrecision(4).toString();
        let str_exp = t.exp.toPrecision(4).toString();
        let props = "";
        let i = 0;
        for (let a of t.attributions) {
            if (i > 0) {
                props += ", ";
                if (i == t.attributions.length - 1) {
                    props += "and ";
                }
            }
            props += a.name;
            i++
        }
        r.predictions.push(`The digit ${t.class} was ${action} with confidence ${str_prob} and explainability ${str_exp} due to the ${props} properties`);
        action = "an alternative";
    }

    // check accuracy of first choice
    if (r.label == r.vote_tally[0].class) {
        correct++;
    } else {
        // TODO looks for other alternatives
        if (r.label == r.vote_tally[1]) {
            second_choice++;
        }
    }
    final_predictions.push(r.vote_tally[0].class);

    count++;
    results.push(r);

}

console.log("==========================================================")
console.log("Final result:", correct / count);
//console.log("Result 1st and 2nd choice:", (correct + second_choice) / count, second_choice);
get_confusion_matrix(final_labels, final_predictions);

fs.writeFileSync(path.join(options.knowledgebase, "results.json"), JSON.stringify(results, null, 4));
fs.writeFileSync(path.join(options.knowledgebase, "kb.json"), JSON.stringify(kb, null, 4));

function get_confusion_matrix(labels, preds, auc) {
    let max = Math.max(...labels);
    let min = Math.min(...labels);
    if (max_label < max) {
        max_label = max;
    }
    if (min_label > min) {
        min_label = min;
    }

    // init the confusion matrix
    let cm = new Array();
    let row_len =  max - min + 1;
    for (let i = 0; i < row_len; i++) {
        cm.push(new Array(row_len).fill(0));
    }
    // init array counting the actual elements per class
    let actuals = new Array(row_len).fill(0);

    // columns are prediction and rows are actual

    let label_count = 0;
    for (let label of labels) {
        actuals[label] += 1;
        let actual = label -  min;
        let pred = preds[label_count] - min;
        cm[actual][pred] += 1;
        label_count++;
    }

    console.log("confusion matrix");
    console.table(cm);

    let s = new Array(row_len).fill().map(u => ({actual: 0, sum: 0, tp: 0, tn: 0, btn: 0, fp: 0, fn: 0, accuracy: 0, cba: 0, sensitivity: 0, specificity: 0, precision: 0, t_product: 0, auc: 0, mcc: 0}));

    for (let j = 0; j < row_len; j++) {
        for (let i = 0; i < row_len; i++) {
            if (j == i) {
                s[j]["tp"] += cm[j][i];
            } else {
                s[j]["fn"] += cm[j][i];
                s[i]["fp"] += cm[j][i]
            }

            for (let k = 0; k < row_len; k++) {
                if (k != j && k != i) {
                    s[k]["tn"] += cm[j][i];
                }
            }

        }
    }

    label_count = 0;
    for (let actual of actuals) {
        let sum = 0;
        for (let key in s[label_count]) {
            sum += s[label_count][key]
        }
        s[label_count]['actual'] = actual;
        s[label_count]['sum'] = sum;
        label_count++;
    }

    let count = 0;
    for (let t of s) {
        // balanced true negative? TODO revisit name
        t.btn = t.tn / (max_label - min_label);
        // number of correct predictions
        t.accuracy = (t.tp + t.tn) / t.sum;
        // CBA class balanced accuracy TODO revisit name TODO problem for unbalanced using avg?
        t.cba = (t.tp + t.btn) / (t.actual + (t.sum / (max_label - min_label + 1)));
        // probability detecting the class given the class
        t.sensitivity = t.tp / t.actual; // same as recall
        // probability of properly detecting not this class
        t.specificity = t.tn / (t.tn + t.fp);
        // out of all positive predictions, ratio that were correct
        if (t.tp > 0 || t.fp > 0) {
            t.precision = t.tp / (t.tp + t.fp);
        }
        // product
        t.t_product = get_product(t);
        if (auc) {
            t.auc = auc[count]
        }
        t.mcc = (t.tp * t.tn - t.fp * t.fn) / Math.sqrt((t.tp + t.fp)*(t.tp+t.fn)*(t.tn+t.fp)*(t.tn+t.fn));
        count++;
    }

    // format the floating point values
    const PREC = 4
    let fmt_s = [];
    for (let t of s) {
        new_t = {}
        for(let n in t) {
            if (t[n] % 1 != 0) {
                new_t[n] = t[n].toPrecision(PREC);
            } else {
                new_t[n] = t[n]
            }
        }
        fmt_s.push(new_t);
    }
    console.log("stats");
    console.table(fmt_s);

    return {
        stats: s,
        confusion_matrix: cm
    };
}

function get_product(stats) {
    return stats.accuracy * stats.sensitivity * stats.specificity * stats.precision;
}

function get_contribution(stats, metrics, metric_name) {

    if (stats === undefined) {
        // TODO change to describe the stat metric used
        console.log("----------Effectiveness contribution is " + metric_name + " ----------");
        return;
    }

    if (metric_name === 'product') {
        return get_product(stats); // product
        //return metrics['train_acc'] * metrics['train_recall'] * metrics['train_specificity'] * metrics['train_precision']; // metrics alternative product
    } else if (metric_name === 's_dot_r') { // specificity * recall
        return metrics['train_specificity'] * metrics['train_recall'];
    } else if (metric_name === 'p_dot_r_dot_s') { // specificity * recall * precision
        return metrics['train_specificity'] * metrics['train_recall'] * metrics['train_precision'];
    } else if (metric_name === 'balanced_acc') {
        return metrics['train_balanced_acc'];
    } else if (metric_name === 'auc') {
        return metrics['train_auc'];
    } else if (metric_name === 'f_score') {
        return metrics['train_f_score'];
    }else if (metric_name === 'balanced_acc_product') { 
        return  metrics['train_recall'] * metrics['train_balanced_acc'] * metrics['train_specificity'] * metrics['train_precision']; // balanced acc product
    }else if (metric_name === 'f_score_product') {
        return metrics['train_f_score'] * metrics['train_acc'] * metrics['train_specificity']; // f score product - f - score = harmonic mean of precision and recall
    } else if (metric_name === 'cba_product') {
        return stats['cba'] * metrics['train_recall'] * metrics['train_specificity'] * metrics['train_precision'];
    } else if (metric_name === 'mcc') {
        return stats.mcc;
    }  else if (metric_name === 'r') {
        return metrics['train_recall']; // original
    } else if (metric_name === 's') {
        return metrics['train_specificity'];
    }  else if (metric_name === 'acc') {
        return metrics['train_acc'];
    }  else if (metric_name === 'p') {
        return metrics['train_precision'];
    } else if (metric_name === 'g_mean') {
        return stats['g_mean']
    } else if (metric_name == 'cohen_kappa') {
        return metrics['cohen_kappa']
    } else if (metric_name == 'p_dot_r') {
        return metrics['train_recall'] * metrics['train_precision'];
    } else if (metric_name == 'p_dot_s') {
        return metrics['train_specificity'] * metrics['train_precision'];
    } else {
        console.log("Invalid metric name,", metric_name);
        process.exit(1);
    } 
}