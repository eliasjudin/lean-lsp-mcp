import EvalPlayground.Basics.Arithmetic
import EvalPlayground.Navigation.Chain

namespace EvalPlayground
namespace Diagnostics

def mismatch : Nat :=
  let base := EvalPlayground.Basics.double 4
  base + missingHelper 3

def inconsistent (n : Nat) : Nat :=
  EvalPlayground.Navigation.composed n + ghostIncrement n

def emptyWitness : False := by
  have : True := trivial
  exact absurd this obviousContradiction

end Diagnostics
end EvalPlayground
