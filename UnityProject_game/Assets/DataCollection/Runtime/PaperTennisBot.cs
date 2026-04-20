using UnityEngine;

namespace TennisDataCollection.Runtime
{
    [RequireComponent(typeof(Collider))]
    public class PaperTennisBot : MonoBehaviour
    {
        [Header("Scene")]
        public Transform ball;
        public Rigidbody ballBody;
        public PaperTennisBot opponent;
        public GameEpisodeRunner runner;
        public Animator animator;
        public Collider strikeCollider;

        [Header("Court")]
        public Vector2 landingXRange = new Vector2(-4.1f, 4.1f);
        public Vector2 landingAbsZRange = new Vector2(0.5f, 11.0f);
        public float groundY = 1.295f;
        public float netZ = 0f;
        public float maxAbsReceiverX = 6.5f;
        public float maxAbsReceiverZ = 18.5f;

        [Header("Shot generation")]
        public Vector2 launchAngleDegRange = new Vector2(10f, 20f);
        public Vector2 preferredSpeedRange = new Vector2(14f, 22f);
        public int launchSampleAttempts = 40;
        public int ballisticSolveAttempts = 12;

        [Header("Net clearance")]
        public float netHeightAboveGround = 1.07f;
        public float netClearanceMargin = 0.15f;
        public float netHalfWidth = 5.8f;

        [Header("Receiver positioning")]
        public Vector2 receiverOffsetRange = new Vector2(5f, 7f);
        public float receiverMoveSpeed = 40f;

        [Header("Hit gating")]
        public float minHitCooldownSec = 0.08f;

        Vector3 homePosition;
        Vector3 moveTarget;
        bool canHit;
        float lastHitTime = -999f;

        public PaperTennisBot Opponent => opponent;
        public Vector3 HomePosition => homePosition;

        void Awake()
        {
            if (ball != null && ballBody == null) ballBody = ball.GetComponent<Rigidbody>();
            if (animator == null) animator = GetComponent<Animator>();
            if (strikeCollider == null) strikeCollider = GetComponent<Collider>();
            homePosition = transform.position;
            moveTarget = homePosition;
        }

        public void ResetForEpisode(Vector3 home)
        {
            homePosition = home;
            moveTarget = home;
            transform.position = home;
            canHit = false;
            lastHitTime = -999f;
        }

        public void ReturnHome()
        {
            canHit = false;
            moveTarget = homePosition;
        }

        public void PrepareIncomingShot(Vector3 landing, bool allowHit)
        {
            canHit = allowHit;
            moveTarget = BuildReceiveTarget(landing);
        }

        public void DisableHitting() => canHit = false;

        public bool LaunchOpeningShot(out Vector3 landing)
        {
            canHit = false;
            return TryLaunchShot(out landing);
        }

        void FixedUpdate()
        {
            Vector3 cur = transform.position;
            Vector3 dst = new Vector3(moveTarget.x, cur.y, moveTarget.z);
            if ((dst - cur).sqrMagnitude <= 0.0001f) return;
            transform.position = Vector3.MoveTowards(cur, dst, receiverMoveSpeed * Time.fixedDeltaTime);
        }

        void OnTriggerEnter(Collider other)
        {
            if (!canHit) return;
            if (!other.CompareTag("Ball")) return;
            if (ballBody == null) return;
            if ((Time.time - lastHitTime) < minHitCooldownSec) return;

            canHit = false;
            if (!TryLaunchShot(out Vector3 landing)) return;
            if (runner != null) runner.OnBotHit(this, landing);
        }

        bool TryLaunchShot(out Vector3 landing)
        {
            landing = transform.position;
            if (ball == null || ballBody == null) return false;

            Vector3 bestLanding = transform.position;
            Vector3 bestLaunch = Vector3.zero;
            float bestClearance = float.NegativeInfinity;
            bool foundCandidate = false;
            bool foundClearLaunch = false;

            for (int i = 0; i < Mathf.Max(1, launchSampleAttempts); i++)
            {
                Vector3 candidateLanding = PickLandingTarget();
                if (!TryComputeLaunchVelocity(ball.position, candidateLanding, out Vector3 candidateLaunch, out float candidateClearance))
                {
                    continue;
                }

                foundCandidate = true;
                if (candidateClearance > bestClearance)
                {
                    bestClearance = candidateClearance;
                    bestLanding = candidateLanding;
                    bestLaunch = candidateLaunch;
                }

                if (candidateClearance >= netClearanceMargin)
                {
                    foundClearLaunch = true;
                    break;
                }
            }

            if (!foundCandidate || !foundClearLaunch)
            {
                if (runner != null)
                {
                    string clearanceText = float.IsFinite(bestClearance) ? bestClearance.ToString("F3") : "none";
                    runner.InvalidateCurrentEpisode($"insufficient_net_clearance:{clearanceText}");
                }
                return false;
            }

            landing = bestLanding;
            PlaySwingAnimation();
            ballBody.linearVelocity = bestLaunch;
            lastHitTime = Time.time;
            return true;
        }

        Vector3 PickLandingTarget()
        {
            float opponentSign = 0f;
            if (opponent != null)
            {
                opponentSign = Mathf.Sign(opponent.homePosition.z - netZ);
            }
            if (opponentSign == 0f)
            {
                opponentSign = -Mathf.Sign(homePosition.z - netZ);
            }
            if (opponentSign == 0f) opponentSign = 1f;

            float x = Random.Range(landingXRange.x, landingXRange.y);
            float z = opponentSign * Random.Range(landingAbsZRange.x, landingAbsZRange.y);
            return new Vector3(x, groundY, z);
        }

        Vector3 BuildReceiveTarget(Vector3 landing)
        {
            float mySideSign = Mathf.Sign(homePosition.z - netZ);
            if (mySideSign == 0f) mySideSign = 1f;

            float offset = Random.Range(receiverOffsetRange.x, receiverOffsetRange.y);
            float x = Mathf.Clamp(landing.x, -maxAbsReceiverX, maxAbsReceiverX);
            float z = Mathf.Clamp(landing.z + mySideSign * offset, -maxAbsReceiverZ, maxAbsReceiverZ);
            return new Vector3(x, homePosition.y, z);
        }

        bool TryComputeLaunchVelocity(Vector3 from, Vector3 to, out Vector3 launch, out float clearance)
        {
            bool foundClearLaunch = false;
            Vector3 bestLaunch = Vector3.zero;
            float bestClearance = float.NegativeInfinity;
            float bestCost = float.PositiveInfinity;

            for (int i = 0; i < Mathf.Max(1, ballisticSolveAttempts); i++)
            {
                float angleDeg = Random.Range(launchAngleDegRange.x, launchAngleDegRange.y);
                float angleRad = angleDeg * Mathf.Deg2Rad;
                if (!TrySolveBallisticSpeed(from, to, angleRad, out float speed))
                {
                    continue;
                }

                Vector3 candidateLaunch = BuildLaunchVector(from, to, angleRad, speed);
                float candidateClearance = EvaluateNetClearance(from, candidateLaunch);
                if (candidateClearance > bestClearance)
                {
                    bestClearance = candidateClearance;
                    bestLaunch = candidateLaunch;
                }

                if (candidateClearance < netClearanceMargin)
                {
                    continue;
                }

                float clamped = Mathf.Clamp(speed, preferredSpeedRange.x, preferredSpeedRange.y);
                float cost = Mathf.Abs(speed - clamped);
                if (cost < bestCost)
                {
                    bestCost = cost;
                    bestLaunch = candidateLaunch;
                    bestClearance = candidateClearance;
                    foundClearLaunch = true;
                    if (cost <= 0.001f)
                    {
                        break;
                    }
                }
            }

            launch = bestLaunch;
            clearance = bestClearance;
            return foundClearLaunch || float.IsFinite(bestClearance);
        }

        static Vector3 BuildLaunchVector(Vector3 from, Vector3 to, float angleRad, float speed)
        {
            Vector3 delta = to - from;
            Vector2 horiz = new Vector2(delta.x, delta.z);
            Vector3 horizDir = new Vector3(horiz.x, 0f, horiz.y).normalized;
            return horizDir * (speed * Mathf.Cos(angleRad)) + Vector3.up * (speed * Mathf.Sin(angleRad));
        }

        static bool TrySolveBallisticSpeed(Vector3 from, Vector3 to, float angleRad, out float speed)
        {
            float g = -Physics.gravity.y;
            Vector3 delta = to - from;
            Vector2 horiz = new Vector2(delta.x, delta.z);
            float dXZ = horiz.magnitude;
            if (dXZ < 0.01f) dXZ = 0.01f;
            float dY = delta.y;

            float tan = Mathf.Tan(angleRad);
            float cos = Mathf.Cos(angleRad);
            float denom = 2f * cos * cos * (dXZ * tan - dY);
            if (denom <= 0.0001f)
            {
                speed = 0f;
                return false;
            }

            float v2 = (g * dXZ * dXZ) / denom;
            if (v2 <= 0f)
            {
                speed = 0f;
                return false;
            }

            speed = Mathf.Sqrt(v2);
            return float.IsFinite(speed);
        }

        float EvaluateNetClearance(Vector3 from, Vector3 launchVelocity)
        {
            if (Mathf.Abs(launchVelocity.z) <= 0.001f)
            {
                return float.NegativeInfinity;
            }

            float timeToNet = (netZ - from.z) / launchVelocity.z;
            if (timeToNet <= 0f)
            {
                return float.NegativeInfinity;
            }

            float xAtNet = from.x + launchVelocity.x * timeToNet;
            if (Mathf.Abs(xAtNet) > netHalfWidth)
            {
                return float.PositiveInfinity;
            }

            float yAtNet = from.y + launchVelocity.y * timeToNet + 0.5f * Physics.gravity.y * timeToNet * timeToNet;
            float netTopY = groundY + netHeightAboveGround;
            return yAtNet - netTopY;
        }

        void PlaySwingAnimation()
        {
            if (animator == null || ball == null) return;
            Vector3 rel = ball.position - transform.position;
            animator.Play(rel.x >= 0f ? "forehand" : "backhand");
        }

        public bool IsBallStrikeable()
        {
            if (ball == null || strikeCollider == null) return false;
            Vector3 closest = strikeCollider.ClosestPoint(ball.position);
            return (closest - ball.position).sqrMagnitude <= 0.0001f;
        }
    }
}
