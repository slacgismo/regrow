from numpy import array
def wecc240_psse(case):

   # dclines
   case["dcline"] = array([
      [168,38,1,2500.0,2450.0,0.0,0.0]
   ])

   return case