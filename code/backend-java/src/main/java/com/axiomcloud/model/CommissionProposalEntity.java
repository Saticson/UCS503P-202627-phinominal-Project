package com.axiomcloud.model;

import jakarta.persistence.*;
import org.hibernate.annotations.CreationTimestamp;

import java.time.LocalDateTime;

@Entity
@Table(name = "commission_proposals")
public class CommissionProposalEntity {

    @Id
    @Column(nullable = false, length = 100)
    private String id;

    @Column(name = "commission_id", nullable = false, length = 100)
    private String commissionId;

    @Column(name = "expert_name", nullable = false, length = 255)
    private String expertName;

    @Column(name = "expert_email", nullable = false, length = 255)
    private String expertEmail;

    @Column(name = "bid_amount", nullable = false, length = 50)
    private String bidAmount;

    @Column(name = "estimated_days")
    private int estimatedDays;

    @Column(name = "proposal_text", columnDefinition = "TEXT")
    private String proposalText;

    @CreationTimestamp
    @Column(name = "created_at", nullable = false, updatable = false)
    private LocalDateTime createdAt;

    public CommissionProposalEntity() {
        this.createdAt = LocalDateTime.now();
    }

    public CommissionProposalEntity(String id, String commissionId, String expertName,
                                    String expertEmail, String bidAmount, int estimatedDays,
                                    String proposalText) {
        this.id = id;
        this.commissionId = commissionId;
        this.expertName = expertName;
        this.expertEmail = expertEmail;
        this.bidAmount = bidAmount;
        this.estimatedDays = estimatedDays;
        this.proposalText = proposalText;
        this.createdAt = LocalDateTime.now();
    }

    // Getters and Setters
    public String getId() { return id; }
    public void setId(String id) { this.id = id; }

    public String getCommissionId() { return commissionId; }
    public void setCommissionId(String commissionId) { this.commissionId = commissionId; }

    public String getExpertName() { return expertName; }
    public void setExpertName(String expertName) { this.expertName = expertName; }

    public String getExpertEmail() { return expertEmail; }
    public void setExpertEmail(String expertEmail) { this.expertEmail = expertEmail; }

    public String getBidAmount() { return bidAmount; }
    public void setBidAmount(String bidAmount) { this.bidAmount = bidAmount; }

    public int getEstimatedDays() { return estimatedDays; }
    public void setEstimatedDays(int estimatedDays) { this.estimatedDays = estimatedDays; }

    public String getProposalText() { return proposalText; }
    public void setProposalText(String proposalText) { this.proposalText = proposalText; }

    public LocalDateTime getCreatedAt() { return createdAt; }
    public void setCreatedAt(LocalDateTime createdAt) { this.createdAt = createdAt; }
}
