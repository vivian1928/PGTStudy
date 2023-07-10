(define (problem sussman)
    (:domain BLOCKS)
    (:objects
        A B C - block
        
        ;;added
        red yellow - colour
    )
    (:init
        (on C A)        
        (clear C)
        (ontable A)
        (ontable B)
        (clear B)
        (handempty)
        
        ;;added
        (colour-of A yellow)
        (colour-of B red)
        (colour-of C red)
        
        (handclean)
    
    )
    (:goal (and
    
    (on B C)
    (on A B)
    ))


)