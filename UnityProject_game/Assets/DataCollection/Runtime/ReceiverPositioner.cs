using UnityEngine;

namespace TennisDataCollection.Runtime
{
    // Sibling component to the tutorial's Bot.cs. Implements the paper's
    // "receiver positions itself 5-7 m behind the predicted bounce" rule
    // (App. B.2) without modifying any tutorial source.
    //
    // Each FixedUpdate we ballistically forward-integrate the ball to
    // y = groundY and, if it will land on this bot's side, set targetZ to
    // (landing.z + sideSign * offset). LateUpdate moves the bot in z toward
    // targetZ at a realistic player speed. We use LateUpdate so the
    // tutorial's Bot.Update->Move() (which writes transform.position based
    // on its own initial z) doesn't overwrite us — LateUpdate runs after
    // every Update, so we have the final word.
    [RequireComponent(typeof(Bot))]
    public class ReceiverPositioner : MonoBehaviour
    {
        public Transform ball;
        public Rigidbody ballBody;
        public Vector2 receiverOffsetRange = new Vector2(5f, 7f);
        public float moveSpeed = 8f;          // realistic sprint
        public float groundY = 1.295f;        // tutorial scene's court surface
        // Tutorial scene's back walls sit at roughly z = ±13. Clamp the
        // dynamic target so the bot never moves past them — otherwise the
        // ball hits a wall and Ball.cs teleports it back. Paper's 5-7 m
        // offset gets honoured when the landing is shallow; for deep
        // landings the offset effectively shrinks.
        public float maxAbsTargetZ = 12.5f;

        float targetZ;
        float homeZ;
        float mySideSign;

        void Start()
        {
            homeZ = transform.position.z;
            mySideSign = Mathf.Sign(homeZ);
            if (mySideSign == 0f) mySideSign = 1f;
            targetZ = homeZ;
        }

        void FixedUpdate()
        {
            if (ball == null || ballBody == null) return;

            Vector3 p = ball.position;
            Vector3 v = ballBody.linearVelocity;

            bool ballOnOpponentSide = Mathf.Sign(p.z) != mySideSign && Mathf.Abs(p.z) > 0.5f;
            bool ballMovingToward   = Mathf.Sign(v.z) == mySideSign && Mathf.Abs(v.z) > 0.5f;

            if (ballOnOpponentSide && ballMovingToward)
            {
                // Solve p.y + v.y t - 0.5 g t^2 = groundY for positive t.
                float g = -Physics.gravity.y;
                float a = -0.5f * g;
                float b = v.y;
                float c = p.y - groundY;
                float disc = b * b - 4f * a * c;
                if (disc >= 0f)
                {
                    float sqrtD = Mathf.Sqrt(disc);
                    float t1 = (-b + sqrtD) / (2f * a);
                    float t2 = (-b - sqrtD) / (2f * a);
                    float t = Mathf.Max(t1, t2);
                    if (t > 0f)
                    {
                        float landingZ = p.z + v.z * t;
                        if (Mathf.Sign(landingZ) == mySideSign)
                        {
                            float offset = Random.Range(receiverOffsetRange.x, receiverOffsetRange.y);
                            float t_z = landingZ + mySideSign * offset;
                            // Clamp inside walls so Ball.cs doesn't teleport.
                            if (mySideSign > 0f)      t_z = Mathf.Min(t_z,  maxAbsTargetZ);
                            else                      t_z = Mathf.Max(t_z, -maxAbsTargetZ);
                            targetZ = t_z;
                        }
                    }
                }
            }
            else if (!ballMovingToward)
            {
                // Ball not heading our way (we just hit it, or it's idle) → drift home.
                targetZ = homeZ;
            }
        }

        void LateUpdate()
        {
            float curZ = transform.position.z;
            float dz = targetZ - curZ;
            float maxStep = moveSpeed * Time.deltaTime;
            if (Mathf.Abs(dz) > maxStep) dz = Mathf.Sign(dz) * maxStep;
            if (dz == 0f) return;
            var p = transform.position;
            p.z = curZ + dz;
            transform.position = p;
        }
    }
}
