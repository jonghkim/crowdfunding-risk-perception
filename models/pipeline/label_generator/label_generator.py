class LabelGenerator:

    def get_label(self, perceived_risk, label_type):

        if label_type == 'numerical_type1':
            perceived_risk = [(risk-3)/float(4) for risk in perceived_risk]
            return perceived_risk

        elif label_type == 'numerical_type2':
            perceived_risk = perceived_risk
            return perceived_risk

        elif label_type == 'categorical_type1':
            label = [1 if risk >3 else 0 for risk in perceived_risk]

            print("Categorical Label Classification Result")
            print("Low Risk: ", len(label)-sum(label))
            print("High Risk: ", sum(label))

            return label
                    
        elif label_type == 'categorical_type2':            
            label =  [1 if risk >=3 else 0 for risk in perceived_risk]    

            print("Categorical Label Classification Result")
            print("Low Risk: ", len(label)-sum(label))
            print("High Risk: ", sum(label))

            return label

        else:
            raise Exception('No Label Type is Matched')