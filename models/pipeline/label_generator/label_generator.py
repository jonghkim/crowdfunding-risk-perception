class LabelGenerator:

    def get_label(self, label_type, perceived_risk):

        if label_type == 'numerical':
            return perceived_risk

        elif label_type == 'categorical_type1':
            label = [1 if risk >3 else 0 for risk in perceived_risk]

            print("Categorical Label Classification Result")
            print("Low Risk: ", (label==0).sum())
            print("High Risk: ",(label==1).sum())

            return label
                    
        elif label_type == 'categorical_type2':            
            label =  [1 if risk >=3 else 0 for risk in perceived_risk]    

            print("Categorical Label Classification Result")
            print("Low Risk: ", (label==0).sum())
            print("High Risk: ",(label==1).sum())

            return label

        else:
            return None