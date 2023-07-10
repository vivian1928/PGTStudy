(define (domain scanalyzer)
  (:requirements :typing) 
  (:types         
        batch segment - object
  )
  (:predicates 
    (on ?b1 - batch ?s1 - segment)    
    (CYCLE-2 ?s1 - segment ?s2 - segment)
    (CYCLE-2-WITH-ANALYSIS ?s1 - segment ?s2 - segment)  
    (analyzed ?b - batch)
  )

  (:action rotate-2
    :parameters (?s1 ?s2 - segment
                 ?b1 ?b2 - batch)
    :precondition (and (CYCLE-2 ?s1 ?s2)
                       (on ?b1 ?s1) (on ?b2 ?s2))
    :effect (and
                (not (on ?b1 ?s1)) (on ?b1 ?s2)
                (not (on ?b2 ?s2)) (on ?b2 ?s1)
            ;    (increase (total-cost) 1)
            )
    )
   
   (:action rotate-and-analyze-2
        :parameters (?s1 ?s2 - segment
                     ?b1 ?b2 - batch)
        :precondition (and
                        (CYCLE-2-WITH-ANALYSIS ?s1 ?s2)
                        (on ?b1 ?s1) (on ?b2 ?s2))
        :effect (and (not (on ?b1 ?s1)) (on ?b1 ?s2)
                     (not (on ?b2 ?s2)) (on ?b2 ?s1)
                     (analyzed ?b1)
                ;     (increase (total-cost) 3)
                )
        
        
    )
)