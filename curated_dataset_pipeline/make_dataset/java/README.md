# Requirement

- antlr-4.13.1-complete.jar 
- python package: antlr4-python3-runtime (if use python3)

Note: version of antlr4 can be different but need to make sure that version of jar file and version of runtime is the same
# Command

```bash
java -jar antlr-4.13.1-complete.jar <YOUR_PARSER> <YOUR_LEXER> -Dlanguage=Python3
```
e.g:
```bash
java -jar antlr-4.13.1-complete.jar JavaParser.g4 JavaLexer.g4 -Dlanguage=Python3
```

This command will create python files that implement java parser with grammar defined in JavaParser.g4 and JavaLexer.g4. Normally, using antlr4 grammar for java, you can get from this [link](https://github.com/antlr/grammars-v4/tree/master/java/java8)