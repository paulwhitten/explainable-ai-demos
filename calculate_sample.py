import json
import pickle
import argparse
import numpy as np
from transform_parallel import TransformNames
from io import BytesIO
import imageio
import base64

# TODO remove skel-fill
# TODO add ability for mixed explainability 

#raw, thresh, skel, fill, corners, ellipse, circle, ellipse_circle, skel_fill, crossings, endpoints, lines, chull

class CalcObject:
    
    def __init__(self, model_folder, kb_folder):
        self.models = {}
        for n in TransformNames:
            model = pickle.load(open(model_folder + "/" + n + ".model", 'rb'))
            self.models[n]= model
        f = open(kb_folder + "/" + "kb.json")
        self.kb = json.load(f)
        f.close()
        self.max_label = max(self.kb["labels"])
        f = open(kb_folder + "/" + "cm.json")
        self.cm = json.load(f)
        f.close()
        # Load explainability scores
        f = open("trans_explainability.json")
        self.explainability = json.load(f)
        f.close()

    def get_failures(self, class_id):
        res = ""
        fp = 0
        mistakes = []
        tp = self.cm[class_id][class_id]
        for ix, row in enumerate(self.cm):
            if ix != class_id:
                fp += self.cm[ix][class_id]
                if self.cm[ix][class_id] > 0:
                    d = {}
                    d["row"] = ix
                    d["count"] = self.cm[ix][class_id]
                    mistakes.append(d)
        fdr = fp / (fp+tp)
        mistakes.sort(key=lambda x: x["count"], reverse=True)
        res += "<br>" + "{:.3g}".format(fdr * 100) + "% of the time that I select a " + str(class_id) + ", I am wrong."
        m_count = 0
        for m in mistakes:
            err_rate = m["count"] / (tp + fp)
            if err_rate > 0.005:
                if m_count == 0:
                    res += "<br>I confuse " + str(m["row"]) + " for " + str(class_id) + " about " + "{:.3g}".format(100*err_rate) + "% of the time." 
                m_count += 1
        return res
    
    def get_result(self, raw, thresh, skel, fill, corner, ellipse, circle, ellipse_circle, skel_fill, crossing, endpoint, line, chull, pca):
        count = 0

        vote_tally = []
        for i in range(self.max_label + 1):
            vote_tally.append({"class": i, "value": 0.0, "explainability": 0.0, "attributions": [], "failures": self.get_failures(i)})

        for n in TransformNames:
            if n == "raw":
                img = raw
            elif n == "thresh":
                img = thresh
            elif n == "skel":
                img = skel
            elif n == "fill":
                img = fill
            elif n == "corner":
                img = corner
            elif n == "ellipse":
                img = ellipse
            elif n == "circle":
                img = circle
            elif n == "ellipse-circle":
                img = ellipse_circle
            elif n == "skel-fill":
                img = skel_fill
            elif n == "crossing":
                img = crossing
            elif n == "endpoint":
                img = endpoint
            elif n == "line":
                img = line
            elif n == "chull":
                img = chull
            elif n == "pca":
                img = pca
            else:
                print("No match for:", n)

            if n != "skel-fill" and n != "thresh": # raw
                print("property:", n)
                # convert img to normalized
                i = np.array(img).flatten().astype('float32')/255

                # was predict([i])
                pred = self.models[n].predict_proba([i])
                class_pred = np.argmax(pred[0])
                print("prediction probability:", pred[0], ", prediction:", class_pred) #pred[0]
 
                #eff = self.kb["stats"][n]["stats"][pred[0]]["accuracy"] * self.kb["stats"][n]["stats"][pred[0]]["sensitivity"] * self.kb["stats"][n]["stats"][pred[0]]["specificity"] * self.kb["stats"][n]["stats"][pred[0]]["precision"]
                eff = self.kb["stats"][n]["stats"][class_pred]["t_product"]

                print("transform:", n, ", prediction:", class_pred, ", effectiveness:", eff, ", probability:", pred[0][class_pred])

                buf = BytesIO()
                print(n, "type:", img.dtype, "min:", np.min(img), "max:", np.max(img))
                
                # Handle different image formats properly
                if img.dtype == bool:
                    # Boolean arrays (like skeleton) - convert directly
                    img = (img * 255).astype(np.uint8)
                elif img.dtype != np.uint8:
                    # Float arrays - check range and convert appropriately
                    if np.max(img) <= 1.0:
                        # Normalized 0-1 range
                        img = (img * 255).astype(np.uint8)
                    else:
                        # Already in 0-255 range
                        img = img.astype(np.uint8)
                
                imageio.imwrite(buf, img, format='png')
                b64_img = base64.b64encode(buf.getbuffer()).decode('utf-8')
                print("base 64 image:", b64_img)

                # Calculate weighted contributions
                contribution = eff * pred[0][class_pred]
                vote_tally[class_pred]["value"] += contribution
                
                # Add explainability-weighted contribution
                explainability_score = self.explainability.get(n, 0.0)  # Get explainability score for this transform
                vote_tally[class_pred]["explainability"] += contribution * explainability_score
                
                vote_tally[class_pred]["attributions"].append({
                    "name": n, 
                    "value": contribution, 
                    "exp": explainability_score,
                    "effectiveness": eff, 
                    "image": b64_img
                }) #img.flatten().tolist()

            #self.kb.stats[n].stats[pred[0]].accuracy, self.kb.stats[n].stats[pred[0]].sensitivity,
            #self.kb.stats[n].stats[pred[0]].specificity, self.kb.stats[n].stats[pred[0]].precision
        print("tally:", vote_tally)
        return vote_tally


# TransformNames = [
#     "crossing",
#     "endpoint",
#     "fill",
#     "skel-fill",
#     "skel",
#     "thresh",
#     "line",
#     "ellipse",
#     "circle",
#     "ellipse-circle",
#     "chull",
#     "raw",
#     "corner"
# ]