(define (domain COLOURBLOCKS)
	(:requirements :strips :typing)
	(:types block hand colour - object)
	(:predicates
		(ontable ?x - block)
		(on ?x - block ?y - block)
		(clear ?x - block)
		(handempty ?h - hand)
		(above ?h - hand ?x - block)
		(holding ?x - block)
		(coloured ?x - block ?c - colour)
		(havecolour ?c - colour) ;which colour's block is above on hand
		(cleaned ?h - hand)
	)
	(:action clean
		:parameters(?c - colour)
		:precondition(and(handempty)(havecolour ?c))
		:effect(and(cleaned)(not (havecolour ?c)))
	)
	(:action pick-up
		:parameters(?x - block ?c - colour)
		:precondition(and(clear ?x)(ontable ?x)(handempty)(cleaned)) ;must be cleaned because of different colour
		:effect(
			and(holding ?x)
			(not (clear ?x))
			(not (ontable ?x))
			(not (handempty))
			(havecolour ?c)
			(not (cleaned))
		)
	)
	(:action pick-up-2
		:parameters(?x - block ?c - colour)
		:precondition(and(clear ?x)(ontable ?x)(handempty)(havecolour ?c)(coloured ?x ?c))
		:effect(
			and(not(clear ?x))
			(not(ontable ?x))
			(holding ?x)
			(not (handempty))
		)
	)
	(:action put-down
		:parameters(?x - block)
		:precondition(holding ?x)
		:effect(
			and(not(holding ?x)
				(handempty)
				(clear ?x)
				(ontable ?x)
			)
		)
	)
	(:action stack
		:parameters(?x - block ?y - block)
		:precondition(and(holding ?x)(clear ?y))
		:effect(
			(and (not (holding ?x))
		   	(not (clear ?y))
		   	(clear ?x)
		   	(handempty)
		   	(on ?x ?y))
		)
	)
	(:action unstack
		:parameters(?x - block ?y - block ?c - colour)
		:precondition(and(handempty)(cleaned)(on ?x ?y)(coloured ?x ?c)(clear ?x))
		:effect(
			(and(not (handempty))
				(not (cleaned))
				(not (clear ?x))
				(havecolour ?c)
				(not (on ?x ?y))
				(cleaned ?y)
				(holding ?x)
			)
		)
	)
	(:action unstack-2
		
	)
)