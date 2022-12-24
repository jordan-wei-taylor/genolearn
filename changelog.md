Version 1.0.2
=============

Bug Fixes
---------

- fixed bad input warning

Misc
----

- switched from tag versioning to branch / tag versioning 


Version 1.0.1
=============

Bug Fixes
---------

- mis-print when user input type does not conform to accepted types

<br>

Version 1.0.0
=============


Enhancements
------------

- added "\*" input to allow either
  - all the remaining choices when the input is "\*" e.g. "\*" -> "{choice1}, {choice2}, ..."
  - take the default and append to string e.g. "\*-example" -> "{default}-example"
- factored code for consistency
- inline warning if wrong user input

Misc
----

- switched from branch to tag versioning

<br>

Version 0.1.2
=============

Bug Fixes
---------

- temp fix for iOS users executing the preprocess sequence / combine commands [issue:#1](https://github.com/jordan-wei-taylor/genolearn/issues/1)


Enhancements
------------

- factored code based on module os
- seperated ``count`` and ``proportion`` subcommands in the ``print metadata`` command
- added binary ``feature selection`` to support ``binary`` flag in ``train`` command


<br>

Version 0.1.1
=============

-  Fixed big in evaluate: Legacy use of the key "Test" instead of "Val" in cli


<br>

Version 0.1.0
=============


-  Removed sparse capabilities as they were not utilised
-  Allow count to binary data conversion to model feature presense instead of feature count
-  Fixed Feature Importance rank for csv
-  Allow to train on entire dataset (built-in to the train command with no additional command)
  
