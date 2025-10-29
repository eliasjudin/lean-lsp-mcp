import EvalPlayground.Basics.Arithmetic
import EvalPlayground.Navigation.Chain

namespace EvalPlayground
namespace Proofs

open Basics Navigation

theorem double_self_ge (n : Nat) : n ≤ Basics.double n := by
  simpa [Basics.double, Nat.add_comm] using Nat.le_add_left n n

example (a b : Nat) :
    Basics.double (a + b) = Basics.double a + Basics.double b := by
  sorry

example (n : Nat) :
    Navigation.handshake n n = Navigation.composed n + Basics.double (n + 2) := by
  rfl

end Proofs
end EvalPlayground
