(define (problem scanalyzer_ex)
    (:domain scanalyzer)
    
    (:objects
        A B C D E F - segment
        b1 b2 b3 b4 b5 b6 - batch
    )
    
    (:init
        ;(= (total-cost) 0)
        (CYCLE-2 A D) (CYCLE-2 A E) (CYCLE-2 A F)
        (CYCLE-2 B D) (CYCLE-2 B E) (CYCLE-2 B F)
        (CYCLE-2 C D) (CYCLE-2 C E) (CYCLE-2 C F)
        (CYCLE-2-WITH-ANALYSIS A F)
        
        (on b1 A) (on b2 B) (on b3 C)
        (on b4 D) (on b5 E) (on b6 F)
    )
    
    (:goal (and
        (analyzed b1) (analyzed b2) (analyzed b3)
        (analyzed b4) (analyzed b5) (analyzed b6)
        (on b1 A) (on b2 B) (on b3 C)
        (on b4 D) (on b5 E) (on b6 F)
    ))


)