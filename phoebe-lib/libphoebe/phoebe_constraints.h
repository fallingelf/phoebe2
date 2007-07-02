#ifndef _PHOEBE_CONSTRAINTS_H
#define _PHOEBE_CONSTRAINTS_H 1

#include "phoebe_parameters.h"

typedef enum {
	PHOEBE_NODE_TYPE_ADD,
	PHOEBE_NODE_TYPE_SUB,
	PHOEBE_NODE_TYPE_MUL,
	PHOEBE_NODE_TYPE_DIV,
	PHOEBE_NODE_TYPE_POT,
	PHOEBE_NODE_TYPE_BUILTIN
} PHOEBE_node_type;

typedef struct PHOEBE_ast {
	enum {
		PHOEBE_AST_NUMVAL,
		PHOEBE_AST_STRING,
		PHOEBE_AST_PARAMETER,
		PHOEBE_AST_NODE
	} type;
	union {
		double numval;
		char  *str;
		PHOEBE_parameter *par;
		struct {
			PHOEBE_node_type        type;
			struct PHOEBE_ast_list *args;
		} node;
	} val;
} PHOEBE_ast;

typedef struct PHOEBE_ast_list {
	PHOEBE_ast *elem;
	struct PHOEBE_ast_list *next;
} PHOEBE_ast_list;

PHOEBE_ast_list *phoebe_ast_construct_list (PHOEBE_ast *ast, PHOEBE_ast_list *list);

PHOEBE_ast *phoebe_ast_add_numval    (double numval);
PHOEBE_ast *phoebe_ast_add_builtin   (char *builtin);
PHOEBE_ast *phoebe_ast_add_parameter (PHOEBE_parameter *par);
PHOEBE_ast *phoebe_ast_add_node      (PHOEBE_node_type type, PHOEBE_ast_list *args);

int phoebe_ast_print (int depth, PHOEBE_ast *in);

typedef struct PHOEBE_constraint {
	PHOEBE_ast *func;
	struct PHOEBE_constraint *next;
} PHOEBE_constraint;

#endif /* _PHOEBE_CONSTRAINTS_H */
