import torch as T
import numpy as np
import random
import torch.nn.functional as F
from reasoner import im

def max_element_difference(tensor1, tensor2):
    if tensor1.shape != tensor2.shape:
        raise ValueError("Tensory muszą mieć ten sam kształt")
    diff = T.abs(tensor1 - tensor2)
    max_diff = T.max(diff).item()
    
    return max_diff

def print_identity_results(trained_reasoner, trained_test_encoders, seed):
    random.seed(seed)
    losses_double_negation = []
    diffs_double_negation = []

    losses_duality = []
    diffs_duality = []

    losses_duality.append( F.l1_loss(trained_reasoner.bot_concept[0], trained_reasoner.not_nn(trained_reasoner.top_concept[0])).item() )
    diffs_duality.append(max_element_difference(trained_reasoner.bot_concept[0], trained_reasoner.not_nn(trained_reasoner.top_concept[0])))

    losses_idempotence = []
    diffs_idempotence = []
    losses_associativity = []
    diffs_associativity = []
    losses_bot_concept_sub =[]
    losses_commutativity = []
    diffs_commutativity = []
    losses_contradiction = []
    diffs_contradiction = []
    losses_identity_top = []
    diffs_identity_top = []
    losses_absorption_bottom = []
    diffs_absorption_bottom = []
    losses_top_subsumption = []
    losses_bot_concept_self = []
    
    one = T.tensor([1.])
    for _ in range(1000):
        encoder = trained_test_encoders[int(np.round(random.random() * (len(trained_test_encoders ) - 1) , 0))]
        input1 = encoder.concepts[int(np.round( random.random() * encoder.n_concepts , 0) - 1) ]
        input2 = encoder.concepts[int(np.round(random. random() * encoder.n_concepts, 0) - 1)]
        input3 = encoder.concepts[int(np.round(random. random() * encoder.n_concepts, 0) - 1)]
        
        double_negation = trained_reasoner.not_nn(trained_reasoner.not_nn(input1))
        losses_double_negation.append( F.l1_loss(input1, double_negation).item() )
        diffs_double_negation.append(max_element_difference(input1, double_negation))


        idempotence = trained_reasoner.and_nn(im(input1, input1))
        losses_idempotence.append( F.l1_loss(input1, idempotence).item() )
        diffs_idempotence.append(max_element_difference(input1, idempotence))

        
        assoc_left = trained_reasoner.and_nn(im(input1, trained_reasoner.and_nn(im(input2, input3))))
        assoc_right = trained_reasoner.and_nn(im(trained_reasoner.and_nn(im(input1, input2)), input3))
        losses_associativity.append(F.l1_loss(assoc_left, assoc_right).item())
        diffs_associativity.append(max_element_difference(assoc_left, assoc_right))
        
        comm_left = trained_reasoner.and_nn(im(input1, input2))
        comm_right = trained_reasoner.and_nn(im(input2, input1))
        losses_commutativity.append(F.l1_loss(comm_left, comm_right).item())
        diffs_commutativity.append(max_element_difference(comm_left, comm_right))
        
        contradiction = trained_reasoner.and_nn(im(input1, trained_reasoner.not_nn(input1)))
        losses_contradiction.append(F.l1_loss(contradiction, trained_reasoner.bot_concept[0]).item())
        diffs_contradiction.append(max_element_difference(contradiction, trained_reasoner.bot_concept[0]))
        
        identity_top = trained_reasoner.and_nn(im(input1, trained_reasoner.top_concept[0]))
        losses_identity_top.append(F.l1_loss(identity_top, input1).item())
        diffs_identity_top.append(max_element_difference(identity_top, input1))
        
        absorption_bottom = trained_reasoner.and_nn(im(input1, trained_reasoner.bot_concept[0]))
        losses_absorption_bottom.append(F.l1_loss(absorption_bottom, trained_reasoner.bot_concept[0]).item())
        diffs_absorption_bottom.append(max_element_difference(absorption_bottom, trained_reasoner.bot_concept[0]))
        
        top_subsumption = trained_reasoner.sub_nn(im(input1, trained_reasoner.top_concept[0]))
        losses_top_subsumption.append(F.binary_cross_entropy_with_logits(top_subsumption, one).item())
        
        losses_bot_concept_sub.append(F.binary_cross_entropy_with_logits(trained_reasoner.sub_nn(im(trained_reasoner.bot_concept[0], input1)), one).item())

        losses_bot_concept_self.append( F.binary_cross_entropy_with_logits(trained_reasoner.sub_nn(im(trained_reasoner.bot_concept[0], trained_reasoner.bot_concept[0])),one).item())


    def print_results(name, losses, diffs):
        print(f"{name}:")
        print(f"  Mean Loss: {np.mean(losses):.6f}")
        print(f"  Std Dev Loss: {np.std(losses):.6f}")
        if diffs:
            print(f"  Mean Diff: {np.mean(diffs):.6f}")
            print(f"  Std Dev Diff: {np.std(diffs):.6f}")
        else:
            print("  Mean Diff: N/A")
            print("  Std Dev Diff: N/A")
        print()

    # Print results for each identity
    print_results("Double Negation", losses_double_negation, diffs_double_negation)
    print_results("Duality of Top and Bottom", losses_duality, diffs_duality)
    print_results("Idempotence", losses_idempotence, diffs_idempotence)
    print_results("Associativity", losses_associativity, diffs_associativity)
    print_results("Commutativity", losses_commutativity, diffs_commutativity)
    print_results("Contradiction", losses_contradiction, diffs_contradiction)
    print_results("Identity with Top", losses_identity_top, diffs_identity_top)
    print_results("Absorption by Bottom", losses_absorption_bottom, diffs_absorption_bottom)
    print_results("Top Concept Subsumption", losses_top_subsumption, [])
    print_results("Bottom Concept Subsumption", losses_bot_concept_sub, [])
    print_results("Bottom Concept Self-Subsumption", losses_bot_concept_self, [])

