1. k_index (k_idx) have to be 1-based
2. kbpl -> kpbl
3. no function call in if branch
4. no boolean field and boolean literals
5. gt4py frontend implements `visitor_*` for ast.py 
6. get function source by `inspect.getsource` in `GTScriptParser`
7. `visitor_With` -> `_visit_computation_node`, `_visit_inteval_node`
8. usable functions inside gt4py: [`ABS`, `MOD`, `SIN`, `COS`, `TAN`, `ARCSIN`, `ARCCOS`, `ARCTAN`, `SQRT`, `EXP`, `LOG`,
    `ISFINITE`, `ISINF`, `ISNAN`, `FLOOR`, `CEIL`, `TRUNC`]
9. can't have temporary var inside if-conditionals
10. clone `serialbox2` from VulcanClimateModeling
11. In the right conda env, build `serialbox2`: `cmake -DCMAKE_INSTALL_PREFIX=/usr/local/serialbox -DSERIALBOX_USE_NETCDF=ON -DSERIALBOX_ENABLE_FORTRAN=ON -DSERIALBOX_TESTING=ON -DSERIALBOX_USE_OPENSSL=OFF ..`
12. best practice for debugging: PyCharm + Docker
13. `dp`, `tem1`, `tem2`, `dv1h`, `rd` ... are not fields
14. [TODO] fix temp vars in part3,4