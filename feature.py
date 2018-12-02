#!/usr/bin/env python3

from androguard.core.bytecodes import apk, dvm
from androguard.core.analysis.analysis import Analysis
import re

def get_permissions(path):
  """
  Get the permissions from an app.

  Parameters:
    path - The path of the app to be decompiled

  Returns:
    A sorted list of permissions

  """

  app = apk.APK(path)
  perms = app.get_permissions()

  # Make sure there is no redundancies, and then sort the list.
  perms = list(set(perms))
  perms.sort()

  return perms


def get_apis(path):
  """
  Get the APIs from an app.

  Parameters:
    path - The path of the app to be decompiled

  Returns:
    A sorted list of APIs with parameters

  """

  # You can see the documents of androguard to get the further details
  # of the decompilation procedures.
  app = apk.APK(path)
  app_dex = dvm.DalvikVMFormat(app.get_dex())
  app_x = Analysis(app_dex)

  methods = set()
  cs = [cc.get_name() for cc in app_dex.get_classes()]

  for method in app_dex.get_methods():
    g = app_x.get_method(method)

    if method.get_code() == None:
      continue

    for i in g.get_basic_blocks().get():
      for ins in i.get_instructions():
        # This is a string that contains methods, variables, or
        # anything else.
        output = ins.get_output()
        
        # Here we use regular expression to check if it is a function
        # call. A function call comprises four parts: a class name, a
        # function name, zero or more parameters, and a return type.
        # The pattern is actually simple:
        # 
        #      CLASS NAME: starts with a character L and ends in a right
        #                  arrow.
        #   FUNCTION NAME: starts with the right arrow and ends in a
        #                  left parenthesis.
        #      PARAMETERS: are between the parentheses.
        #     RETURN TYPE: is the rest of the string.
        #
        match = re.search(r'(L[^;]*;)->[^\(]*\([^\)]*\).*', output)
        if match and match.group(1) not in cs:
          methods.add(match.group())

  methods = list(methods)
  methods.sort()

  return methods


def main():
  """
  For test

  """

if __name__ == '__main__':
  main()

