import EvalPlayground.Basics.Arithmetic

namespace EvalPlayground
namespace Navigation

def composed (n : Nat) : Nat :=
  Basics.offset (n + 1) + Basics.triple n

def handshake (a b : Nat) : Nat :=
  composed a + Basics.double (b + 2)

def balance (a b : Nat) : Nat :=
  Basics.cascade a b + composed b

end Navigation
end EvalPlayground
